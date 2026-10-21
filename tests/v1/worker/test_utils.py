# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import ctypes.util
import mmap
import threading
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch

import vllm.v1.attention.backends.mla.index_group as index_group_module
import vllm.v1.hisparse.runtime as hisparse_runtime_module
from vllm.config import CacheConfig, CUDAGraphMode
from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.distributed.kv_transfer import kv_transfer_state
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse import (
    worker as hisparse_worker_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
    HiSparseConnectorWorker,
    _flatten_row_mirrors,
    _select_written_row_mirrors,
    _SlotMappingStaging,
)
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import MemorySnapshot
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.hisparse.runtime import HiSparseCacheHandle
from vllm.v1.hisparse.types import SparseKVPageTransfer, SparseKVRowMirror
from vllm.v1.worker.utils import (
    bind_kv_cache,
    bind_kv_cache_to_layers,
    copy_kv_cache_blocks_inplace,
    request_memory,
)


def _make_hisparse_worker() -> HiSparseConnectorWorker:
    worker = object.__new__(HiSparseConnectorWorker)
    worker._slot_mapping_staging = None
    worker._row_mirror_num_rows = 0
    worker._per_layer_mirrored = set()
    worker._submitted_mirror_layers = set()
    worker._draft_layers = ()
    worker._draft_mirror_pending = False
    worker._pending_dma_descriptors = deque()
    worker._dma_free_descriptors = []
    worker.host_write_events = (MagicMock(), MagicMock())
    worker.host_write_event = worker.host_write_events[1]
    worker._next_host_write_event = 0
    worker.dma_stream = None
    worker.shared_host_region = None
    worker._completed_host_copy_dst_ids = []
    worker._metrics_calls = 0
    worker._metrics_event = MagicMock()
    worker._metrics_pending = False
    worker.leader_runtimes = []
    return worker


def _as_cache_handles(*handles: SimpleNamespace) -> list[HiSparseCacheHandle]:
    """Hand the worker SimpleNamespace stand-ins for its cache handles.

    Each stand-in carries only the HiSparseCacheHandle fields its test's code
    path reads, so reading any other field still raises AttributeError.
    """
    return [*handles]  # type: ignore[list-item]  # SimpleNamespace stand-ins


def test_hisparse_worker_get_kv_connector_stats_reads_completed_snapshot(monkeypatch):
    worker = _make_hisparse_worker()
    worker._metrics_calls = hisparse_worker_module._METRICS_INTERVAL - 1
    worker._metrics_pending = False
    worker._metrics_event = MagicMock()
    worker._metrics_event.query.return_value = True
    compute_stream = MagicMock()
    group = SimpleNamespace(
        swap_stats=torch.tensor([12, 4], dtype=torch.uint64),
        swap_stats_host=torch.empty(2, dtype=torch.uint64),
        stats_row_bytes=16,
        copy_stream=MagicMock(),
    )
    worker.leader_runtimes = [SimpleNamespace(index_group=group)]  # type: ignore[list-item]  # runtime stand-in
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        hisparse_worker_module, "current_stream", lambda: compute_stream
    )

    assert worker.get_kv_connector_stats() is None
    compute_stream.wait_stream.assert_called_once_with(group.copy_stream)
    group.copy_stream.wait_stream.assert_called_once_with(compute_stream)
    worker._metrics_event.record.assert_called_once_with()
    assert group.swap_stats.tolist() == [0, 0]

    stats = worker.get_kv_connector_stats()
    assert stats is not None
    assert stats.data == {
        "cache_hits": [12],
        "cache_misses": [4],
        "host_to_device_bytes": [64],
        "host_cache_usage_perc": [],
        "pending_page_transfers": [],
        "host_block_lifetime_seconds": [],
        "host_block_idle_before_evict_seconds": [],
        "host_block_reuse_gap_seconds": [],
    }


def test_hisparse_row_mirrors_follow_runner_request_order():
    first = SparseKVRowMirror((1,), 10, 1)
    second = SparseKVRowMirror((2,), 20, 1)
    row_mirrors = {"first": (first,), "second": (second,)}

    assert _flatten_row_mirrors(row_mirrors, ["second", "first"]) == (
        second,
        first,
    )


def test_hisparse_gpu_slots_select_written_rows_from_async_envelope():
    """Rejected speculative rows must never become durable host KV."""
    candidates = (
        SparseKVRowMirror((100, 500), 1000, 4),
        SparseKVRowMirror((20, 300), 2000, 4),
    )
    source_slots = np.array([102, 103, -1, 21, 22, 23], dtype=np.int64)

    assert _select_written_row_mirrors(candidates, source_slots, 0) == (
        SparseKVRowMirror((102, 502), 1002, 2),
        SparseKVRowMirror((21, 301), 2001, 3),
    )


def test_hisparse_written_rows_preserve_scheduler_page_boundaries():
    candidates = (
        SparseKVRowMirror((0,), 100, 4),
        SparseKVRowMirror((4,), 104, 4),
    )

    assert _select_written_row_mirrors(
        candidates, np.array([2, 3, 4, 5], dtype=np.int64), 0
    ) == (
        SparseKVRowMirror((2,), 102, 2),
        SparseKVRowMirror((4,), 104, 2),
    )


def test_hisparse_written_rows_without_host_destinations_are_skipped():
    assert (
        _select_written_row_mirrors((), np.array([100, 101], dtype=np.int64), 0) == ()
    )


def test_hisparse_appends_reference_slots_within_a_mirror_phase(monkeypatch):
    """Separate context and query writes must share one mirror phase."""
    worker = _make_hisparse_worker()
    worker.is_host_writer = True
    state = _SlotMappingStaging(
        MagicMock(), MagicMock(), torch.empty(4, dtype=torch.int64)
    )
    worker._slot_mapping_staging = state
    handle = SimpleNamespace(runtime=SimpleNamespace(resident_source_index=1))
    worker.cache_layer_names = ["layer"]
    worker.cache_handles = _as_cache_handles(handle)
    worker._row_mirror_num_rows = 1
    worker._submitted_mirror_layers = set()
    main_stream = MagicMock()
    monkeypatch.setattr(hisparse_worker_module, "current_stream", lambda: main_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    handle.slot_mapping = torch.tensor([7, 8], dtype=torch.int64)
    worker._stage_row_mirror_mapping(2)
    handle.slot_mapping = torch.tensor([9, 10], dtype=torch.int64)
    worker._stage_row_mirror_mapping(2)

    torch.testing.assert_close(state.slots, torch.tensor([7, 8, 9, 10]))
    assert state.stream.wait_stream.call_args_list == [call(main_stream)] * 2
    assert state.event.record.call_args_list == [call(state.stream)] * 2
    assert state.num_tokens == 4
    assert state.source_index == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA cache binding")
@pytest.mark.parametrize(
    ("mode", "expected_calls"),
    [(CUDAGraphMode.PIECEWISE, 1), (CUDAGraphMode.FULL, 0)],
)
def test_hisparse_submits_layer_mirror_at_replayed_attention_boundary(
    mode, expected_calls, dist_init
):
    """Piecewise replay submits real DMA; FULL leaves it for the batch fallback."""
    from tests.v1.attention.test_sparse_mla_backends import (
        _bound_hisparse_worker,
        _hisparse_mirror_metadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
        HiSparseConnectorMetadata,
    )
    from vllm.forward_context import set_forward_context

    slots = torch.tensor([2, 3], dtype=torch.int64, device="cuda")
    resident = torch.arange(4 * 4 * 4, dtype=torch.float32, device="cuda").view(4, 4, 4)
    worker = _bound_hisparse_worker((resident,), slot_mapping=slots)
    try:
        kv_transfer_state.ensure_kv_transfer_initialized(
            worker.vllm_config, worker.kv_cache_config
        )
        handle = worker.cache_handles[0]
        handle.runtime.host_cache.zero_()
        metadata = _hisparse_mirror_metadata(slots, decode=False)
        worker.start_step(
            HiSparseConnectorMetadata(
                command=None,
                host_block_copies=(),
                source_block_ids=(),
                row_mirrors={"request": (SparseKVRowMirror((2,), 6, 2),)},
                all_context_pages_resident=True,
                residency_updates={},
            ),
            worker.request_state_indices,
            ["request"],
            num_tokens=2,
        )
        worker.prepare_forward({"layer.0": metadata})
        with patch.object(
            worker, "_enqueue_row_dma", wraps=worker._enqueue_row_dma
        ) as submit:
            with set_forward_context(
                {"layer.0": metadata},
                worker.vllm_config,
                num_tokens=2,
                cudagraph_runtime_mode=mode,
            ):
                handle.finish_kv_update()
            assert submit.call_count == expected_calls
        if expected_calls:
            worker.host_write_event.synchronize()
            torch.testing.assert_close(
                handle.runtime.host_cache[6:8], resident.view(-1, 4)[2:4].cpu()
            )
            assert worker._submitted_mirror_layers == {0}
        else:
            assert not worker._dma_submitted
            assert not worker._submitted_mirror_layers
            assert torch.count_nonzero(handle.runtime.host_cache).item() == 0
    finally:
        worker.shutdown()
        kv_transfer_state.ensure_kv_transfer_shutdown()


def _hisparse_parallel_config(**overrides):
    values = {
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "world_size": 2,
        "distributed_executor_backend": "mp",
        "nnodes_within_dp": 1,
        "data_parallel_index": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_hisparse_shares_host_pool_only_for_local_tp(monkeypatch):
    monkeypatch.setattr(
        hisparse_runtime_module.current_platform, "is_cuda_alike", lambda: True
    )
    config = SimpleNamespace(parallel_config=_hisparse_parallel_config())

    assert hisparse_runtime_module.use_shared_hisparse_host_pool(config)

    unsupported = (
        {"tensor_parallel_size": 1, "world_size": 1},
        {"pipeline_parallel_size": 2, "world_size": 4},
        {"prefill_context_parallel_size": 2, "world_size": 4},
        {"decode_context_parallel_size": 2},
        {"world_size": 4},
        {"distributed_executor_backend": "ray"},
        {"nnodes_within_dp": 2},
    )
    for overrides in unsupported:
        config.parallel_config = _hisparse_parallel_config(**overrides)
        assert not hisparse_runtime_module.use_shared_hisparse_host_pool(config)


@pytest.mark.skip_global_cleanup
def test_hisparse_shared_host_pool_uses_one_replicated_mmap(monkeypatch):
    page = mmap.PAGESIZE
    tp_group = MagicMock(rank_in_group=1, world_size=2)
    monkeypatch.setattr(hisparse_runtime_module, "get_tp_group", lambda: tp_group)

    class FakeSharedOffloadRegion:
        BLOCK_SIZE_ALIGNMENT = page

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.total_size_bytes = kwargs["num_chunks"] * kwargs["kv_bytes_per_chunk"]
            self.base_tensor = torch.empty(self.total_size_bytes, dtype=torch.int8)
            self.is_pinned = False
            self.pinned_addresses = []
            self.view_sizes = []
            self.offset = 0
            kwargs["barrier"]()

        def create_next_canonical_view(self, size):
            self.view_sizes.append(size)
            view = torch.as_strided(
                self.base_tensor,
                size=(self.kwargs["num_chunks"], size),
                stride=(self.kwargs["kv_bytes_per_chunk"], 1),
                storage_offset=self.offset,
            )
            self.offset += size
            return view

        def cleanup(self):
            pass

    monkeypatch.setattr(
        hisparse_runtime_module, "SharedOffloadRegion", FakeSharedOffloadRegion
    )
    registration_ranges = MagicMock(return_value=((0, page), (page, 4 * page)))
    monkeypatch.setattr(
        hisparse_runtime_module,
        "_hisparse_registration_ranges",
        registration_ranges,
    )
    pinned: list[torch.Tensor] = []

    def pin_tensor(tensor):
        tp_group.barrier.assert_called_once_with()
        pinned.append(tensor)

    monkeypatch.setattr(hisparse_runtime_module, "pin_tensor", pin_tensor)
    config = SimpleNamespace(
        instance_id="instance",
        parallel_config=_hisparse_parallel_config(),
    )

    pools, private_pools, region = hisparse_runtime_module.allocate_hisparse_host_pools(
        config,
        [24, 40],
        num_blocks=4,
        host_block_stride=page,
        use_shared_host_pool=True,
    )

    assert region is not None
    assert isinstance(region, FakeSharedOffloadRegion)
    assert private_pools == []
    assert region.kwargs == {
        "engine_id": "hisparse_instance_dp3",
        "num_chunks": 1,
        "rank": 0,
        "kv_bytes_per_chunk": 4 * page,
        "cpu_page_size": 64,
        "barrier": tp_group.barrier,
        "creator_memory_check": hisparse_runtime_module.check_hisparse_host_memory,
        "populate_shard": (1, 2),
    }
    assert region.view_sizes == [24, 40]
    assert [pool.shape for pool in pools] == [(24,), (40,)]
    registration_ranges.assert_called_once_with([24, 40], 4, page)
    assert [
        (tensor.data_ptr() - region.base_tensor.data_ptr(), tensor.nbytes)
        for tensor in pinned
    ] == [(0, page), (page, 3 * page)]
    assert region.is_pinned


@pytest.mark.parametrize("fail_registration", [None, 1, 2])
def test_shared_host_pool_tracks_successful_registrations(
    monkeypatch, fail_registration
):
    """Only successful registrations may be unregistered on allocation failure."""
    page = mmap.PAGESIZE
    tp_group = MagicMock(rank_in_group=0, world_size=2)
    monkeypatch.setattr(hisparse_runtime_module, "get_tp_group", lambda: tp_group)
    backing = torch.empty(3 * page, dtype=torch.uint8)
    region = MagicMock(base_tensor=backing, pinned_addresses=[], is_pinned=False)
    monkeypatch.setattr(
        hisparse_runtime_module, "SharedOffloadRegion", lambda **kw: region
    )
    monkeypatch.setattr(
        hisparse_runtime_module,
        "_hisparse_registration_ranges",
        lambda *args: ((0, page), (page, 3 * page)),
    )
    cudart = MagicMock()
    cudart.cudaHostRegister.side_effect = [
        SimpleNamespace(value=int(fail_registration == i)) for i in (1, 2)
    ]
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    config = SimpleNamespace(
        instance_id="test", parallel_config=SimpleNamespace(data_parallel_index=0)
    )
    context = (
        pytest.raises(RuntimeError, match="cudaHostRegister failed")
        if fail_registration
        else nullcontext()
    )
    with context:
        hisparse_runtime_module.allocate_hisparse_host_pools(
            config, [3 * page], 3, page, use_shared_host_pool=True
        )
    successful = 2 if fail_registration is None else fail_registration - 1
    assert region.pinned_addresses == [
        backing.data_ptr() + i * page for i in range(successful)
    ]
    assert region.is_pinned == bool(successful)
    assert region.cleanup.call_count == int(fail_registration is not None)
    attempts = 2 if fail_registration is None else fail_registration
    assert (
        cudart.cudaHostRegister.call_args_list
        == [
            call(backing.data_ptr(), page, 0),
            call(backing.data_ptr() + page, 2 * page, 0),
        ][:attempts]
    )


def test_shared_host_pool_registers_layer_spans_in_one_backing(monkeypatch):
    """Aliased tensor configs must not count the shared backing once per view."""
    region = SimpleNamespace(base_tensor=torch.empty(64, dtype=torch.int8))
    allocate = MagicMock(return_value=([], [], region))
    monkeypatch.setattr(
        hisparse_runtime_module, "allocate_hisparse_host_pools", allocate
    )
    config = SimpleNamespace(
        hisparse_shared_host_pool=True,
        hisparse_host_num_blocks=4,
        hisparse_host_block_stride=16,
        kv_cache_tensors=[
            SimpleNamespace(
                host_resident=True,
                layers=["a", "b"],
                offset=0,
                layer_stride=12,
                size=44,
            ),
            SimpleNamespace(
                host_resident=True, layers=["c"], offset=24, layer_stride=20, size=44
            ),
        ],
    )
    vllm_config = SimpleNamespace()
    pool = hisparse_runtime_module.HiSparseHostPool(vllm_config, config)
    backing = pool.allocate(44)
    allocate.assert_called_once_with(
        vllm_config, [12, 12, 20], 4, 16, use_shared_host_pool=True
    )
    assert backing.numel() == 44
    assert backing.data_ptr() == region.base_tensor.data_ptr()
    assert pool.registered is region.base_tensor


def test_hisparse_registration_chunks_end_between_host_blocks():
    """Registration seams must not bisect a DMA-addressable host block."""
    page = mmap.PAGESIZE
    ranges = hisparse_runtime_module._hisparse_registration_ranges(
        tensor_sizes=[4 * 3 * page, 4 * 5 * page],
        num_blocks=4,
        host_block_stride=8 * page,
        max_chunk_bytes=10 * page,
    )

    assert ranges == (
        (0, 9 * page),
        (9 * page, 17 * page),
        (17 * page, 27 * page),
        (27 * page, 32 * page),
    )


def test_copy_cpu_kv_cache_logical_blocks_ignores_storage_padding():
    backing = torch.full((10, 2, 3), -1, dtype=torch.float32)
    cache = backing[1:9]
    cache[2:4] = 7
    cache[6:8] = 11

    copy_kv_cache_blocks_inplace(
        [cache],
        num_blocks=4,
        kv_cache_block_copies=[
            KVCacheBlockCopy(1, 0),
            KVCacheBlockCopy(3, 2),
        ],
    )

    torch.testing.assert_close(cache[0:2], torch.full_like(cache[0:2], 7))
    torch.testing.assert_close(cache[4:6], torch.full_like(cache[4:6], 11))
    assert (backing[0] == -1).all()
    assert (backing[9] == -1).all()


def test_bind_kv_cache_keeps_scheduler_block_views_for_runner():
    """Block copies index scheduler blocks, so the runner keeps the unmapped
    views while a packed layer is bound with kernel blocks."""

    class _Layer:
        def bind_kv_cache(self, kv_cache):
            self.kv_cache = kv_cache

    num_blocks, row = 4, 64
    storage = torch.zeros(num_blocks * row)
    # Two layers packed into one block row: an 8x4 page, then a 4x2 page
    # split into two 2-row kernel blocks.
    mla = storage.as_strided((num_blocks, 8, 4), (row, 4, 1))
    indexer = storage.as_strided((num_blocks, 1, 4, 2), (row, 8, 2, 1), 40)
    # The indexer in 2-row kernel blocks: block b's kernel block j is b * 16 + j.
    mapped = indexer.as_strided((50, 1, 2, 2), (4, 8, 2, 1))
    kv_caches = {"layers.0.attn": mla, "layers.1.indexer": indexer}
    ctx = {name: _Layer() for name in kv_caches}
    runner_kv_caches: list[torch.Tensor] = []

    bind_kv_cache(
        kv_caches,
        ctx,
        runner_kv_caches,
        layer_kv_caches={**kv_caches, "layers.1.indexer": mapped},
    )

    assert ctx["layers.1.indexer"].kv_cache is mapped
    assert runner_kv_caches == [mla, indexer]
    copies = [KVCacheBlockCopy(1, 3)]
    with pytest.raises(AssertionError, match="not divisible"):
        copy_kv_cache_blocks_inplace([mapped], num_blocks, copies)
    storage[row : 2 * row] = 7
    copy_kv_cache_blocks_inplace(runner_kv_caches, num_blocks, copies)
    assert (mla[3] == 7).all() and (indexer[3] == 7).all()


@pytest.mark.parametrize("shared,rank", [(False, 0), (True, 0), (True, 1)])
@pytest.mark.parametrize("has_copies", [False, True])
def test_hisparse_host_copy_waits_for_writes_on_writer_only(
    monkeypatch, shared, rank, has_copies
):
    """Host CoW must read completed writes; shared non-writers only barrier."""
    worker = _make_hisparse_worker()
    cache = torch.zeros((2, 1, 1))
    worker.host_caches = [cache]
    worker.host_num_blocks = 2
    worker.shared_host_region = object() if shared else None
    event = MagicMock()
    event.synchronize.side_effect = lambda: cache[0].fill_(7)
    barrier = MagicMock()
    monkeypatch.setattr(
        hisparse_worker_module, "get_tensor_model_parallel_rank", lambda: rank
    )
    monkeypatch.setattr(
        hisparse_worker_module, "get_tp_group", lambda: SimpleNamespace(barrier=barrier)
    )
    copies = [KVCacheBlockCopy(0, 1)] if has_copies else []

    worker._copy_host_blocks(copies, event)

    writes = has_copies and (not shared or rank == 0)
    assert cache[1].item() == (7 if writes else 0)
    assert event.synchronize.call_count == int(writes)
    assert barrier.call_count == int(shared and has_copies)
    assert worker._completed_host_copy_dst_ids == ([1] if has_copies else [])


def test_hisparse_worker_updates_request_state_mapping_in_place(monkeypatch):
    worker = _make_hisparse_worker()
    worker.request_state_indices = torch.arange(4, dtype=torch.int32)
    worker._pending_invalid_block_ids = [5]
    invalidations = []
    monkeypatch.setattr(
        worker,
        "invalidate_blocks",
        lambda blocks, states: invalidations.append((blocks.copy(), states.clone())),
    )
    original_ptr = worker.request_state_indices.data_ptr()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    worker.set_request_state_indices(torch.tensor([3, 1], dtype=torch.int32))

    assert worker.request_state_indices.data_ptr() == original_ptr
    assert worker.request_state_indices.tolist() == [3, 1, -1, -1]
    assert len(invalidations) == 1
    assert invalidations[0][0] == [5]
    torch.testing.assert_close(
        invalidations[0][1], torch.tensor([3, 1], dtype=torch.int32)
    )
    assert worker._pending_invalid_block_ids == []


def test_hisparse_pre_forward_transfer_builds_page_descriptors(monkeypatch):
    worker = _make_hisparse_worker()
    worker.is_host_writer = True
    worker.kernel_block_size = 2
    source = torch.empty((3, 2, 4), dtype=torch.uint8)
    destination = torch.empty((6, 4), dtype=torch.uint8)
    worker.resident_caches = (source,)
    worker.host_caches = (destination,)
    worker.cache_handles = _as_cache_handles(
        SimpleNamespace(runtime=SimpleNamespace(resident_source_index=0))
    )
    worker._dma_free_descriptors = []
    submit_dma_descriptors = MagicMock()
    monkeypatch.setattr(worker, "_submit_dma_descriptors", submit_dma_descriptors)

    worker._enqueue_transfers([SparseKVPageTransfer(7, 2, (1,), after_forward=False)])

    descriptors, count = submit_dma_descriptors.call_args.args
    assert count == 1
    assert descriptors.src[:count].tolist() == [
        source.data_ptr() + source.stride(0) * source.element_size()
    ]
    assert descriptors.dst[:count].tolist() == [destination.data_ptr() + 4 * 4]
    assert descriptors.sizes[:count].tolist() == [2 * 4]
    assert submit_dma_descriptors.call_args.kwargs["transfer_ids"] == (7,)


def test_hisparse_eager_mirror_records_transfer_without_page_copy(monkeypatch):
    worker = _make_hisparse_worker()
    worker.cache_handles = _as_cache_handles(
        SimpleNamespace(runtime=SimpleNamespace(eager_host_mirror=True))
    )
    record_transfer_completion = MagicMock()
    enqueue_transfers = MagicMock()
    monkeypatch.setattr(
        worker, "_record_transfer_completion", record_transfer_completion
    )
    monkeypatch.setattr(worker, "_enqueue_transfers", enqueue_transfers)
    transfers = [SparseKVPageTransfer(7, 2, (1,), after_forward=False)]

    worker._submit_transfers(transfers)

    record_transfer_completion.assert_called_once_with(transfers)
    enqueue_transfers.assert_not_called()


@pytest.mark.parametrize("is_host_writer", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_hisparse_tail_restore_preserves_imported_rows(
    monkeypatch, request, is_host_writer, device
):
    """Restore externally pinned host rows on every rank under sync checking."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the device copy check")
    worker = _make_hisparse_worker()
    worker.is_host_writer = is_host_writer
    worker.kernel_block_size = 4
    worker.host_caches = tuple(
        torch.arange(24, dtype=torch.float32).reshape(12, 2) + layer * 100
        for layer in range(2)
    )
    if device == "cuda":
        # Match production's cudaHostRegister pool, which torch.is_pinned()
        # does not recognize, rather than torch's caching pinned allocator.
        host_pool, registered = hisparse_runtime_module.allocate_pinned_host_pool(
            sum(cache.nbytes for cache in worker.host_caches)
        )
        request.addfinalizer(
            lambda: hisparse_runtime_module.release_pinned_state([], [registered])
        )
        host_caches = host_pool.view(torch.float32).view(2, 12, 2).unbind()
        for destination, source in zip(host_caches, worker.host_caches):
            destination.copy_(source)
        worker.host_caches = host_caches
    worker.resident_caches = tuple(
        torch.full((4, 4, 2), -1.0, device=device) for _ in range(2)
    )
    worker.cache_handles = _as_cache_handles(
        *(
            SimpleNamespace(
                runtime=SimpleNamespace(resident_source_index=i, eager_host_mirror=True)
            )
            for i in range(2)
        )
    )
    record_transfer_completion = MagicMock()
    monkeypatch.setattr(
        worker, "_record_transfer_completion", record_transfer_completion
    )
    stream = torch.cuda.current_stream() if device == "cuda" else MagicMock()
    monkeypatch.setattr(hisparse_worker_module, "current_stream", lambda: stream)
    transfer = SparseKVPageTransfer(7, 2, (1, 3), False, restore=True)

    restore_pages = worker._restore_pages
    if device == "cuda":
        import vllm.utils.gpu_sync_debug as gpu_sync_debug

        monkeypatch.setattr(gpu_sync_debug, "_SYNC_CHECK_MODE", "error")
        monkeypatch.setattr(gpu_sync_debug, "_sync_check_enabled", False)
        gpu_sync_debug.enable_gpu_sync_check()
        restore_pages = gpu_sync_debug.with_gpu_sync_check(restore_pages)
    restore_pages([transfer])

    # Appending into unused capacity must preserve every imported prompt row.
    for cache, host, block in zip(
        worker.resident_caches, worker.host_caches, transfer.resident_block_ids
    ):
        assert torch.equal(cache[block].cpu(), host[8:12])
        cache[block, 2:] = 999
        assert torch.equal(cache[block, :2].cpu(), host[8:10])
        assert torch.all(cache[0] == -1)
    record_transfer_completion.assert_called_once_with([transfer], stream=stream)


def test_hisparse_lazy_mirror_copies_transfer_pages(monkeypatch):
    worker = _make_hisparse_worker()
    worker.cache_handles = _as_cache_handles(
        SimpleNamespace(runtime=SimpleNamespace(eager_host_mirror=False))
    )
    record_transfer_completion = MagicMock()
    enqueue_transfers = MagicMock()
    monkeypatch.setattr(
        worker, "_record_transfer_completion", record_transfer_completion
    )
    monkeypatch.setattr(worker, "_enqueue_transfers", enqueue_transfers)
    transfers = [SparseKVPageTransfer(7, 2, (1,), after_forward=False)]

    worker._submit_transfers(transfers)

    enqueue_transfers.assert_called_once_with(transfers)
    record_transfer_completion.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA DMA")
def test_hisparse_dma_row_mirror_builds_descriptors():
    from tests.v1.attention.test_sparse_mla_backends import _bound_hisparse_worker

    source = torch.arange(3 * 2 * 16, dtype=torch.uint8, device="cuda").view(3, 2, 16)
    worker = _bound_hisparse_worker((source,), host_num_rows=6)
    destination = worker.host_caches[0]
    try:
        worker._set_row_mirrors((SparseKVRowMirror((3,), 4, 1),))
        with patch.object(
            worker, "_submit_dma_descriptors", wraps=worker._submit_dma_descriptors
        ) as submit:
            worker._enqueue_row_dma(range(1))
            descriptors, count = submit.call_args.args
            assert count == 1
            assert descriptors.src[:count].tolist() == [source.data_ptr() + 3 * 16]
            assert descriptors.dst[:count].tolist() == [destination.data_ptr() + 4 * 16]
            assert descriptors.sizes[:count].tolist() == [16]
        assert worker._dma_submitted
        assert len(worker._pending_dma_descriptors) == 1
        worker.host_write_event.synchronize()
        assert worker._pending_dma_descriptors[0][0].query()
        torch.testing.assert_close(destination[4], source.view(-1, 16)[3].cpu())
    finally:
        worker.shutdown()


def _dma_host_gate(stream):
    """Block a real CUDA stream at a host callback until the test releases it."""
    release, entered, expired = threading.Event(), threading.Event(), threading.Event()
    callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

    @callback_type
    def callback(_):
        # A CUDA host callback must not call CUDA APIs, including synchronization.
        entered.set()
        if not release.wait(timeout=30):
            expired.set()

    candidates = (
        ctypes.util.find_library("cudart"),
        "libcudart.so.13",
        "libcudart.so.12",
    )
    for name in candidates:
        if name:
            try:
                library = ctypes.CDLL(name)
                break
            except OSError:
                pass
    else:
        raise RuntimeError("CUDA runtime library is required for the DMA event gate")
    launch = library.cudaLaunchHostFunc
    launch.argtypes = (ctypes.c_void_p, callback_type, ctypes.c_void_p)
    launch.restype = ctypes.c_int
    assert launch(ctypes.c_void_p(stream.cuda_stream), callback, None) == 0
    # The caller retains the callback and library through stream completion.
    return release, entered, expired, callback, library


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA DMA")
@pytest.mark.parametrize("layers", [(0, 1, 2), (2, 0), (1,), (2, 2)])
@pytest.mark.parametrize("rebind", [False, True])
def test_hisparse_row_dma_uses_resident_spans(layers, rebind):
    from tests.v1.attention.test_sparse_mla_backends import _bound_hisparse_worker
    from tests.v1.core.test_kv_cache_planning import _hisparse_planning_config
    from vllm.config import SpeculativeConfig
    from vllm.v1.hisparse.runtime import allocate_pinned_host_pool

    sources = (
        torch.empty((4, 4, 16), dtype=torch.uint8, device="cuda"),
        torch.empty((4, 4, 8), dtype=torch.float32, device="cuda"),
        torch.empty_strided(
            (4, 4, 24), (128, 24, 1), dtype=torch.bfloat16, device="cuda"
        ),
    )
    for index, source in enumerate(sources):
        source.copy_(
            torch.arange(source.numel(), device="cuda").view(source.shape) % 97 + index
        )
    config = _hisparse_planning_config()
    # Speculative layouts intentionally use per-leader invalidation. This test
    # exercises DMA only and needs no pointer-table allocation while it is gated.
    config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=1
    )
    worker = _bound_hisparse_worker(sources, config=config, source_indices=(0, 1, 0))
    gate = None
    try:
        rebound = None
        old_sources, old_destinations = sources, worker.host_caches
        if rebind:
            rebound = (
                torch.empty_strided(
                    (5, 4, 16), (80, 16, 1), dtype=torch.uint8, device="cuda"
                ),
                torch.empty_strided(
                    (5, 4, 8), (40, 8, 1), dtype=torch.float32, device="cuda"
                ),
                torch.empty_strided(
                    (5, 4, 24), (160, 24, 1), dtype=torch.bfloat16, device="cuda"
                ),
            )
            new_hosts = []
            for index, source in enumerate(rebound):
                source.copy_(
                    torch.arange(source.numel(), device="cuda").view(source.shape) % 89
                    + 10
                    + index
                )
                host, registered = allocate_pinned_host_pool(
                    18 * source.shape[-1] * source.element_size()
                )
                worker.pinned_host_pools.append(registered)
                new_hosts.append(
                    (host.view(source.dtype).view(18, source.shape[-1]), registered)
                )
        # Prime two real submissions and drain them before the gate. The test
        # does not inject descriptor objects into the worker's private pools.
        worker._set_row_mirrors(
            (SparseKVRowMirror((1, 6), 8, 2), SparseKVRowMirror((8, 1), 2, 1))
        )
        worker._enqueue_row_dma(layers)
        worker._enqueue_row_dma(layers)
        worker.host_write_event.synchronize()
        worker._release_completed_dma_descriptors()
        assert len(worker._dma_free_descriptors) == 2
        assert not worker._pending_dma_descriptors
        for destination in old_destinations:
            destination.zero_()
        torch.accelerator.synchronize()
        if rebind:
            gate = _dma_host_gate(worker.dma_stream)
            assert gate[1].wait(timeout=5), "CUDA host callback did not start"
        worker._set_row_mirrors(
            (SparseKVRowMirror((1, 6), 8, 2), SparseKVRowMirror((8, 1), 2, 1))
        )
        with patch.object(
            worker, "_submit_dma_descriptors", wraps=worker._submit_dma_descriptors
        ) as submit:
            worker._enqueue_row_dma(layers)
            previous, _ = submit.call_args.args
            previous_src = previous.src.clone()
            if rebind:
                pending_event = worker._pending_dma_descriptors[0][0]
                assert not pending_event.query()
                assert rebound is not None
                sources = rebound
                for handle, source, (destination, registered) in zip(
                    worker.cache_handles, sources, new_hosts, strict=True
                ):
                    raw = torch.empty(0, dtype=torch.uint8, device="cuda").set_(
                        source.untyped_storage(),
                        0,
                        (source.untyped_storage().nbytes(),),
                        (1,),
                    )
                    # KV dtype/row width belong to the real runtime; a legal
                    # rebind changes storage, block stride and capacity only.
                    handle.runtime.bind_source_cache(
                        destination, registered_host_pool=registered
                    )
                    handle.bind_cache(
                        raw,
                        byte_offset=0,
                        block_stride=source.stride(0) * source.element_size(),
                        num_blocks=source.shape[0],
                        block_size=4,
                        block_table=handle.block_table,
                        slot_mapping=handle.slot_mapping,
                    )
                worker._init_dma()
                worker._set_row_mirrors(
                    (SparseKVRowMirror((1, 6), 8, 2), SparseKVRowMirror((16, 1), 17, 1))
                )
                worker._release_completed_dma_descriptors()
                assert all(
                    item is not previous for item in worker._dma_free_descriptors
                )
                worker._enqueue_row_dma(layers)
                torch.testing.assert_close(previous.src, previous_src)
                assert not pending_event.query()
            descriptors, count = submit.call_args.args
            assert count == 2 * len(layers)
            source_rows = ((1, 6, 1), (16, 1, 16) if rebind else (8, 1, 8))
            destination_rows = (8, 17 if rebind else 2)
            counts = (2, 1)
            destinations = worker.host_caches
            assert descriptors.src[:count].tolist() == [
                sources[layer].data_ptr()
                + (row[layer] // 4)
                * sources[layer].stride(0)
                * sources[layer].element_size()
                + (row[layer] % 4)
                * sources[layer].shape[-1]
                * sources[layer].element_size()
                for row in source_rows
                for layer in layers
            ]
            assert descriptors.dst[:count].tolist() == [
                destinations[layer].data_ptr()
                + row
                * destinations[layer].shape[-1]
                * destinations[layer].element_size()
                for row in destination_rows
                for layer in layers
            ]
            assert descriptors.sizes[:count].tolist() == [
                rows * sources[layer].shape[-1] * sources[layer].element_size()
                for rows in counts
                for layer in layers
            ]
        if gate is not None:
            gate[0].set()
        worker.host_write_event.synchronize()
        worker._release_completed_dma_descriptors()
        if rebind:
            assert gate is not None
            assert pending_event.query()
            assert any(item is previous for item in worker._dma_free_descriptors)
            assert not gate[2].is_set(), "CUDA host callback timed out"
        if rebind:
            for row, destination, rows in (((1, 6, 1), 8, 2), ((8, 1, 8), 2, 1)):
                for layer in set(layers):
                    block, offset = divmod(row[layer], 4)
                    torch.testing.assert_close(
                        old_destinations[layer][destination : destination + rows],
                        old_sources[layer][block, offset : offset + rows].cpu(),
                    )
        for mirror_index, row in enumerate(source_rows):
            for layer in set(layers):
                block, offset = divmod(row[layer], 4)
                dest = destination_rows[mirror_index]
                torch.testing.assert_close(
                    destinations[layer][dest : dest + counts[mirror_index]],
                    sources[layer][block, offset : offset + counts[mirror_index]].cpu(),
                )
    finally:
        if gate is not None:
            gate[0].set()
        worker.shutdown()


@pytest.mark.parametrize(
    "source_index,source_start,destination_start,count,error",
    [
        (1, 0, 0, 1, "source index"),
        (0, -1, 0, 1, "source is"),
        (0, 12, 0, 1, "source is"),
        (0, 3, 0, 2, "source is"),
        (0, 0, -1, 1, "index is"),
        (0, 0, 8, 1, "index is"),
        (0, 0, 7, 2, "index is"),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA DMA")
def test_hisparse_row_dma_rejects_invalid_span_before_submit(
    source_index, source_start, destination_start, count, error
):
    from tests.v1.attention.test_sparse_mla_backends import _bound_hisparse_worker

    sources = tuple(
        torch.zeros((3, 4, 4), dtype=torch.float32, device="cuda") for _ in range(2)
    )
    worker = _bound_hisparse_worker(sources, host_num_rows=8)
    try:
        worker._set_row_mirrors(
            (SparseKVRowMirror((source_start,), destination_start, count),)
        )
        with patch.object(
            worker, "_submit_dma_descriptors", wraps=worker._submit_dma_descriptors
        ) as submit:
            with pytest.raises(RuntimeError, match=error):
                worker._enqueue_row_dma((source_index,))
            submit.assert_not_called()
        assert not worker._dma_submitted
        assert not worker._pending_dma_descriptors
    finally:
        worker.shutdown()


def test_hisparse_finish_forward_mirrors_all_layers_once(monkeypatch):
    dst_slots = torch.tensor([7, 8, 9], dtype=torch.int64)
    req_ids = torch.tensor([0, 1, 2], dtype=torch.int32)
    leader = SimpleNamespace(
        eager_host_mirror=True,
        is_group_leader=True,
        invalidate_written_slots=MagicMock(),
    )
    follower = SimpleNamespace(
        eager_host_mirror=True,
        is_group_leader=False,
        invalidate_written_slots=MagicMock(),
    )
    handles = [
        SimpleNamespace(
            runtime=runtime,
            decode_batch=True,
            host_mirror_required=True,
            num_actual_tokens=3,
            num_decode_tokens=2,
            req_id_per_token=req_ids,
            mirror_slot_mapping=dst_slots,
        )
        for runtime in (leader, follower)
    ]
    worker = _make_hisparse_worker()
    worker.is_host_writer = True
    worker.cache_handles = handles
    worker.hot_backing = SimpleNamespace(device=torch.device("cuda:0"))
    worker._set_row_mirrors((SparseKVRowMirror((0, 0), 7, 3),))
    worker._dma_submitted = False
    worker._per_layer_mirrored = set()
    worker._submitted_mirror_layers = set()
    worker._post_forward_transfers = []
    enqueue_row_dma = MagicMock()
    monkeypatch.setattr(worker, "_enqueue_row_dma", enqueue_row_dma)
    worker.host_write_event = MagicMock()
    worker._forward_ready_event = MagicMock()
    current_stream = MagicMock()
    monkeypatch.setattr(
        hisparse_worker_module, "current_stream", lambda: current_stream
    )

    worker.finish_forward()

    worker._forward_ready_event.record.assert_called_once_with()
    enqueue_row_dma.assert_called_once_with(
        (0, 1), ready_event=worker._forward_ready_event
    )
    leader.invalidate_written_slots.assert_called_once()
    torch.testing.assert_close(
        leader.invalidate_written_slots.call_args.args[0], dst_slots
    )
    follower.invalidate_written_slots.assert_not_called()
    worker.host_write_event.record.assert_called_once_with(current_stream)


def test_hisparse_finish_forward_does_not_repeat_per_layer_mirrors(monkeypatch):
    slots = torch.tensor([7, 8], dtype=torch.int64)
    runtime = SimpleNamespace(
        eager_host_mirror=False,
        is_group_leader=False,
        invalidate_written_slots=MagicMock(),
    )
    handles = [
        SimpleNamespace(
            runtime=runtime,
            decode_batch=False,
            host_mirror_required=True,
            num_actual_tokens=2,
            num_decode_tokens=0,
            req_id_per_token=torch.empty(0, dtype=torch.int32),
            mirror_slot_mapping=slots,
        )
        for _ in range(2)
    ]
    worker = _make_hisparse_worker()
    worker.is_host_writer = True
    worker.cache_handles = handles
    worker._set_row_mirrors((SparseKVRowMirror((0, 0), 7, 2),))
    worker._per_layer_mirrored = {0, 1}
    worker._submitted_mirror_layers = {0, 1}
    enqueue_row_dma = MagicMock()
    monkeypatch.setattr(worker, "_enqueue_row_dma", enqueue_row_dma)

    worker._enqueue_host_mirror()

    enqueue_row_dma.assert_not_called()


def test_hisparse_prefill_mirrors_source_groups_and_flushes_partial_group(
    monkeypatch,
):
    slots = torch.tensor([7, 8], dtype=torch.int64)
    source_indices = [0, 0, 1, 1, 1, 1, 2]
    handles = [
        SimpleNamespace(
            runtime=SimpleNamespace(
                eager_host_mirror=False,
                is_group_leader=False,
                resident_source_index=source_index,
            ),
            decode_batch=False,
            host_mirror_required=layer_index < 5,
            num_actual_tokens=2 if layer_index < 5 else 0,
            num_decode_tokens=0,
            req_id_per_token=torch.empty(0, dtype=torch.int32),
            mirror_slot_mapping=slots,
        )
        for layer_index, source_index in enumerate(source_indices)
    ]
    worker = _make_hisparse_worker()
    worker.is_host_writer = True
    worker.cache_handles = handles
    worker._set_row_mirrors((SparseKVRowMirror((0, 0, 0), 7, 2),))
    worker._per_layer_mirrored = set()
    worker._submitted_mirror_layers = set()
    worker._layer_ready_events = tuple(MagicMock() for _ in handles)
    enqueue_row_dma = MagicMock()
    monkeypatch.setattr(worker, "_enqueue_row_dma", enqueue_row_dma)

    for layer_index in range(5):
        worker._enqueue_layer_mirror(layer_index)

    enqueue_row_dma.assert_called_once_with(
        (0, 1), ready_event=worker._layer_ready_events[1]
    )
    worker._enqueue_host_mirror(ready_event=worker._layer_ready_events[-1])
    assert enqueue_row_dma.call_args_list[1].args == ((2, 3, 4),)
    assert enqueue_row_dma.call_args_list[1].kwargs == {
        "ready_event": worker._layer_ready_events[-1]
    }


def test_hisparse_finish_forward_rejects_partial_per_layer_mirror(monkeypatch):
    slots = torch.tensor([7, 8], dtype=torch.int64)
    runtime = SimpleNamespace(
        eager_host_mirror=False,
        is_group_leader=False,
        invalidate_written_slots=MagicMock(),
    )
    handles = [
        SimpleNamespace(
            runtime=runtime,
            decode_batch=False,
            host_mirror_required=True,
            num_actual_tokens=2,
            num_decode_tokens=0,
            req_id_per_token=torch.empty(0, dtype=torch.int32),
            mirror_slot_mapping=slots,
        )
        for _ in range(2)
    ]
    worker = _make_hisparse_worker()
    worker.is_host_writer = True
    worker.cache_handles = handles
    worker._set_row_mirrors((SparseKVRowMirror((0, 0), 7, 2),))
    worker._per_layer_mirrored = {0}
    worker._submitted_mirror_layers = set()
    monkeypatch.setattr(worker, "_enqueue_row_dma", MagicMock())

    with pytest.raises(RuntimeError, match="did not mirror every active layer"):
        worker._enqueue_host_mirror()


def test_hisparse_finish_forward_orders_next_forward_after_dma(monkeypatch):
    current_stream = MagicMock()
    worker = _make_hisparse_worker()
    worker.hot_backing = SimpleNamespace(device=torch.device("cuda:0"))
    worker.is_host_writer = True
    worker._dma_submitted = True
    worker.host_write_event = MagicMock()
    worker.cache_handles = _as_cache_handles(
        SimpleNamespace(
            runtime=SimpleNamespace(eager_host_mirror=True), num_actual_tokens=0
        )
    )
    worker._post_forward_transfers = []
    worker._pending_dma_descriptors = deque()
    worker._dma_free_descriptors = []
    worker._forward_ready_event = MagicMock()
    monkeypatch.setattr(worker, "_finish_mirror_phase", MagicMock())
    monkeypatch.setattr(
        hisparse_worker_module, "current_stream", lambda: current_stream
    )

    worker.finish_forward()

    current_stream.wait_event.assert_called_once_with(worker.host_write_event)
    assert not worker._dma_submitted


def test_hisparse_finish_forward_mirrors_standalone_mtp_cache(monkeypatch):
    dst_slots = torch.tensor([7, 8], dtype=torch.int64)
    runtime = SimpleNamespace(
        eager_host_mirror=True,
        is_group_leader=False,
        invalidate_written_slots=MagicMock(),
    )
    inactive = SimpleNamespace(
        runtime=runtime,
        decode_batch=False,
        host_mirror_required=False,
        num_actual_tokens=0,
        num_decode_tokens=0,
        req_id_per_token=None,
        mirror_slot_mapping=dst_slots,
    )
    mtp = SimpleNamespace(
        runtime=runtime,
        decode_batch=True,
        host_mirror_required=True,
        num_actual_tokens=2,
        num_decode_tokens=2,
        req_id_per_token=torch.tensor([0, 1], dtype=torch.int32),
        mirror_slot_mapping=dst_slots,
    )
    worker = _make_hisparse_worker()
    worker.is_host_writer = True
    worker.cache_handles = _as_cache_handles(inactive, mtp)
    worker._set_row_mirrors((SparseKVRowMirror((0,), 7, 2),))
    worker._per_layer_mirrored = set()
    worker._submitted_mirror_layers = set()
    enqueue_row_dma = MagicMock()
    monkeypatch.setattr(worker, "_enqueue_row_dma", enqueue_row_dma)

    worker._enqueue_host_mirror()

    enqueue_row_dma.assert_called_once_with((1,), ready_event=None)


def test_hisparse_shared_host_reader_skips_mirror(monkeypatch):
    """A non-writer TP rank must not mirror rows into the shared host pool."""
    dst_slots = torch.tensor([7, 8], dtype=torch.int64)
    leader = SimpleNamespace(
        eager_host_mirror=True,
        is_group_leader=True,
        invalidate_written_slots=MagicMock(),
    )
    handle = SimpleNamespace(
        runtime=leader,
        decode_batch=True,
        host_mirror_required=True,
        num_actual_tokens=2,
        num_decode_tokens=1,
        req_id_per_token=torch.tensor([0], dtype=torch.int32),
        mirror_slot_mapping=dst_slots,
    )
    worker = _make_hisparse_worker()
    worker.is_host_writer = False
    worker.cache_handles = _as_cache_handles(handle)
    worker._set_row_mirrors((SparseKVRowMirror((0,), 7, 2),))
    worker._per_layer_mirrored = set()
    enqueue_row_dma = MagicMock()
    monkeypatch.setattr(worker, "_enqueue_row_dma", enqueue_row_dma)

    worker._enqueue_host_mirror()

    enqueue_row_dma.assert_not_called()
    leader.invalidate_written_slots.assert_called_once()


def test_hisparse_shared_host_reader_skips_transfer_completion():
    """A non-writer TP rank must not acknowledge host writes."""
    worker = _make_hisparse_worker()
    worker.is_host_writer = False
    worker.kernel_block_size = 1
    worker._enqueued_transfer_ids = []
    worker._pending_transfer_events = []

    worker._record_transfer_completion([SparseKVPageTransfer(1, 2, (3,), True)])

    assert worker._enqueued_transfer_ids == []
    assert worker._pending_transfer_events == []


def test_hisparse_writer_records_transfer_completion_after_dma(monkeypatch):
    worker = _make_hisparse_worker()
    worker.is_host_writer = True
    worker.dma_stream = MagicMock()
    worker._enqueued_transfer_ids = []
    worker._pending_transfer_events = deque()
    event = MagicMock()
    monkeypatch.setattr(torch, "Event", lambda: event)

    worker._record_transfer_completion(
        [
            SparseKVPageTransfer(3, 2, (1,), False),
            SparseKVPageTransfer(7, 4, (5,), True),
        ]
    )

    event.record.assert_called_once_with(worker.dma_stream)
    assert worker._enqueued_transfer_ids == [3, 7]
    assert worker._pending_transfer_events == deque([(event, (3, 7))])


@pytest.mark.parametrize("is_host_writer", [False, True])
def test_hisparse_step_waits_for_previous_host_write(monkeypatch, is_host_writer):
    worker = _make_hisparse_worker()
    worker.is_host_writer = is_host_writer
    worker.hot_backing = SimpleNamespace(device=torch.device("cuda:1"))
    host_write_events = (MagicMock(), MagicMock())
    worker.host_write_events = host_write_events
    worker.host_write_event = host_write_events[1]
    worker._next_host_write_event = 0
    worker.host_caches = ()
    worker.host_num_blocks = 1
    worker._post_forward_transfers = []
    worker._pending_invalid_block_ids = []
    worker.cache_handles = _as_cache_handles(
        SimpleNamespace(
            runtime=SimpleNamespace(eager_host_mirror=True),
            decode_batch=False,
            host_mirror_required=False,
            num_actual_tokens=0,
            num_decode_tokens=0,
            req_id_per_token=None,
        )
    )
    worker._per_layer_mirrored = set()
    worker._submitted_mirror_layers = set()
    stream = MagicMock()
    monkeypatch.setattr(hisparse_worker_module, "current_stream", lambda: stream)

    worker.start_step(
        SimpleNamespace(
            host_block_copies=[],
            command=None,
            source_block_ids=[],
            row_mirrors={},
            all_context_pages_resident=True,
            row_mirrors_from_resident=False,
            residency_updates={},
        ),
        None,
    )
    worker.start_step(
        SimpleNamespace(
            host_block_copies=[],
            command=None,
            source_block_ids=[],
            row_mirrors={},
            all_context_pages_resident=True,
            row_mirrors_from_resident=False,
            residency_updates={},
        ),
        None,
    )

    assert stream.wait_event.call_args_list == [
        call(host_write_events[1]),
        call(host_write_events[0]),
    ]
    assert worker.host_write_event is host_write_events[1]


@pytest.mark.parametrize("tp_rank, expected_copies", [(0, 1), (1, 0)])
def test_hisparse_shared_host_block_copy_has_one_writer(
    monkeypatch, tp_rank, expected_copies
):
    worker = _make_hisparse_worker()
    worker.shared_host_region = object()
    worker.host_caches = (torch.empty(4, 1),)
    worker.host_num_blocks = 4
    previous_event = MagicMock()
    copies = (KVCacheBlockCopy(0, 1),)
    copy_blocks = MagicMock()
    tp_group = MagicMock()
    monkeypatch.setattr(
        hisparse_worker_module, "get_tensor_model_parallel_rank", lambda: tp_rank
    )
    monkeypatch.setattr(hisparse_worker_module, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        hisparse_worker_module, "copy_kv_cache_blocks_inplace", copy_blocks
    )

    worker._copy_host_blocks(copies, previous_event)

    assert copy_blocks.call_count == expected_copies
    tp_group.barrier.assert_called_once_with()
    assert worker.take_completed_host_copies() == [1]
    assert worker.take_completed_host_copies() == []


def test_hisparse_empty_step_does_not_replay_stale_host_mirror(monkeypatch):
    handle = SimpleNamespace(
        runtime=SimpleNamespace(eager_host_mirror=True),
        decode_batch=True,
        num_actual_tokens=2,
        num_decode_tokens=2,
        req_id_per_token=torch.tensor([0, 1]),
        mirror_slot_mapping=torch.tensor([4, 5]),
    )
    worker = _make_hisparse_worker()
    worker.is_host_writer = True
    worker.hot_backing = SimpleNamespace(device=torch.device("cpu"))
    worker.host_write_event = MagicMock()
    worker._forward_ready_event = MagicMock()
    worker.host_caches = ()
    worker.host_num_blocks = 1
    worker.cache_handles = _as_cache_handles(handle)
    worker._per_layer_mirrored = set()
    worker._submitted_mirror_layers = set()
    worker._post_forward_transfers = []
    worker._pending_invalid_block_ids = []
    enqueue_host_mirror = MagicMock(wraps=worker._enqueue_host_mirror)
    monkeypatch.setattr(worker, "_enqueue_host_mirror", enqueue_host_mirror)
    stream = MagicMock()
    monkeypatch.setattr(hisparse_worker_module, "current_stream", lambda: stream)

    worker.start_step(
        SimpleNamespace(
            host_block_copies=[],
            command=None,
            source_block_ids=[],
            row_mirrors={},
            all_context_pages_resident=True,
            row_mirrors_from_resident=False,
            residency_updates={},
        ),
        None,
    )
    worker.finish_forward()

    enqueue_host_mirror.assert_called_once_with(worker._forward_ready_event)
    assert handle.num_actual_tokens == 0
    torch.testing.assert_close(handle.mirror_slot_mapping, torch.tensor([4, 5]))


def test_hisparse_cache_handles_join_index_groups_during_construction(monkeypatch):
    """Followers must not allocate duplicate runtime state before profiling."""
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=2,
            async_scheduling=False,
        ),
        speculative_config=None,
        kv_transfer_config=None,
    )
    resolved = hisparse_runtime_module.ResolvedHiSparseConfig(
        top_k=4,
        device_buffer_size=8,
        max_union_rows=8,
    )
    monkeypatch.setattr(hisparse_runtime_module, "_has_hisparse_ops", lambda: True)
    monkeypatch.setattr(
        hisparse_runtime_module.ResolvedHiSparseConfig,
        "from_vllm_config",
        classmethod(lambda cls, vllm_config, model_top_k: resolved),
    )
    shared_states: list[object] = []
    streams: list[object] = []

    def create_shared_state(_device, _max_rows, _top_k):
        shared_states.append(object())
        return shared_states[-1]

    def create_stream(_device):
        streams.append(object())
        return streams[-1]

    monkeypatch.setattr(
        hisparse_runtime_module, "_create_shared_topk_state", create_shared_state
    )
    monkeypatch.setattr(hisparse_runtime_module, "_create_copy_stream", create_stream)
    monkeypatch.setattr(index_group_module, "_create_side_stream", lambda _: object())
    monkeypatch.setattr(index_group_module, "_create_event", lambda: object())
    index_group_builder = index_group_module.SparseMLAIndexGroupBuilder(
        torch.empty((2, 4), dtype=torch.int32)
    )

    def make_cache_handle(is_leader: bool):
        index_group, _ = index_group_builder.register_layer(is_leader)
        cache_handle = hisparse_runtime_module.create_hisparse_cache_handle(
            config,
            model_top_k=4,
            is_index_group_leader=is_leader,
            row_width=8,
            kv_dtype=torch.float32,
            index_group=index_group,
            device="cpu",
        )
        assert cache_handle is not None
        return cache_handle

    first_leader = make_cache_handle(True)
    first_index_group = index_group_builder.current_group
    first_follower = make_cache_handle(False)
    second_leader = make_cache_handle(True)
    second_index_group = index_group_builder.current_group
    second_follower = make_cache_handle(False)

    assert first_index_group is not None
    assert second_index_group is not None
    assert first_follower.runtime.index_group is first_leader.runtime.index_group
    assert second_follower.runtime.index_group is second_leader.runtime.index_group
    assert first_leader.runtime.index_group is not second_leader.runtime.index_group
    assert first_leader.runtime.index_group.copy_stream is first_index_group.side_stream
    assert (
        first_leader.runtime.index_group.logical_topk_ready
        is first_index_group.logical_topk_ready
    )
    assert (
        second_leader.runtime.index_group.copy_stream is second_index_group.side_stream
    )
    assert (
        second_leader.runtime.index_group.logical_topk_ready
        is second_index_group.logical_topk_ready
    )
    assert len(shared_states) == 2
    assert streams == []


def test_hisparse_worker_shutdown_releases_pinned_state(monkeypatch):
    worker = object.__new__(HiSparseConnectorWorker)
    worker._initialized = True
    worker._slot_mapping_staging = None
    worker.dma_stream = None
    worker.cache_handles = []
    worker.pinned_host_pools = []
    worker.shared_host_region = object()
    worker.vllm_config = SimpleNamespace(instance_id="shutdown")
    pool = hisparse_runtime_module.get_hisparse_host_pool(worker.vllm_config, None)
    released = False

    def release_pinned_state(runtimes, pinned_host_pools, shared_host_region):
        nonlocal released
        assert runtimes == []
        assert pinned_host_pools == []
        assert shared_host_region is worker.shared_host_region
        released = True

    monkeypatch.setattr(
        hisparse_worker_module, "release_pinned_state", release_pinned_state
    )

    worker.shutdown()

    assert released
    new_pool = hisparse_runtime_module.get_hisparse_host_pool(worker.vllm_config, None)
    assert new_pool is not pool
    hisparse_runtime_module.drop_hisparse_host_pool(worker.vllm_config)


class _TestReplaySSMMixer(MambaMixer2):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.use_replayssm = True
        self.mamba_config = MambaConfig(backend=MambaBackendEnum.FLASHINFER)
        self._replayssm_ring_start = torch.empty(0, dtype=torch.int32)
        self._replayssm_prev_num_accepted = torch.empty(0, dtype=torch.int32)
        self._replayssm_prev_query_len = torch.empty(0, dtype=torch.int32)
        self._commits_replayssm_trackers = True
        self._updates_replayssm_trackers = True

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return ((2,), (3,), (4,), (5,), (6,))

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return (torch.float32,) * 5


def _packed_replayssm_cache(num_blocks: int) -> torch.Tensor:
    return torch.full((num_blocks, 1, 1, 80), 0, dtype=torch.int8)


@pytest.mark.parametrize("layers_only", [False, True])
def test_bind_kv_cache_shares_replayssm_trackers_by_cache_group(layers_only):
    mixers = [_TestReplaySSMMixer() for _ in range(3)]
    layer_names = [f"layers.{i}.mixer" for i in range(3)]
    ctx = dict(zip(layer_names, mixers))
    # Reverse insertion order: updater must follow layer index, not dict order.
    kv_cache = {
        layer_names[2]: _packed_replayssm_cache(4),
        layer_names[1]: _packed_replayssm_cache(4),
        layer_names[0]: _packed_replayssm_cache(4),
    }
    kv_cache_groups = [
        SimpleNamespace(layer_names=[layer_names[0], layer_names[2]]),
        SimpleNamespace(layer_names=[layer_names[1]]),
    ]

    if layers_only:
        bind_kv_cache_to_layers(kv_cache, ctx, kv_cache_groups=kv_cache_groups)
    else:
        bind_kv_cache(kv_cache, ctx, [], kv_cache_groups=kv_cache_groups)

    tracker_names = (
        "_replayssm_ring_start",
        "_replayssm_prev_num_accepted",
        "_replayssm_prev_query_len",
    )
    for tracker_name in tracker_names:
        group_tracker = getattr(mixers[0], tracker_name)
        assert group_tracker.data_ptr() == getattr(mixers[2], tracker_name).data_ptr()
        assert group_tracker.data_ptr() != getattr(mixers[1], tracker_name).data_ptr()
        assert group_tracker.shape == (4,)
        assert torch.count_nonzero(group_tracker) == 0

    assert [m._commits_replayssm_trackers for m in mixers] == [True, True, False]
    assert [m._updates_replayssm_trackers for m in mixers] == [False, True, True]


def test_bind_kv_cache(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    ctx = {
        "layers.0.self_attn": Attention(32, 128, 0.1, prefix="layers.0.self_attn"),
        "layers.1.self_attn": Attention(32, 128, 0.1, prefix="layers.1.self_attn"),
        "layers.2.self_attn": Attention(32, 128, 0.1, prefix="layers.2.self_attn"),
        "layers.3.self_attn": Attention(32, 128, 0.1, prefix="layers.3.self_attn"),
    }
    kv_cache = {
        "layers.0.self_attn": torch.zeros((1,)),
        "layers.1.self_attn": torch.zeros((1,)),
        "layers.2.self_attn": torch.zeros((1,)),
        "layers.3.self_attn": torch.zeros((1,)),
    }
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)
    assert ctx["layers.0.self_attn"].kv_cache is kv_cache["layers.0.self_attn"]
    assert ctx["layers.1.self_attn"].kv_cache is kv_cache["layers.1.self_attn"]
    assert ctx["layers.2.self_attn"].kv_cache is kv_cache["layers.2.self_attn"]
    assert ctx["layers.3.self_attn"].kv_cache is kv_cache["layers.3.self_attn"]

    assert runner_kv_caches[0] is kv_cache["layers.0.self_attn"]
    assert runner_kv_caches[1] is kv_cache["layers.1.self_attn"]
    assert runner_kv_caches[2] is kv_cache["layers.2.self_attn"]
    assert runner_kv_caches[3] is kv_cache["layers.3.self_attn"]


def test_bind_kv_cache_non_attention(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    # example from Jamba PP=2
    ctx = {
        "model.layers.20.attn": Attention(32, 128, 0.1, prefix="model.layers.20.attn"),
        "model.layers.28.attn": Attention(32, 128, 0.1, prefix="model.layers.28.attn"),
    }
    kv_cache = {
        "model.layers.20.attn": torch.zeros((1,)),
        "model.layers.28.attn": torch.zeros((1,)),
    }

    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["model.layers.20.attn"].kv_cache is kv_cache["model.layers.20.attn"]
    assert ctx["model.layers.28.attn"].kv_cache is kv_cache["model.layers.28.attn"]

    assert runner_kv_caches[0] is kv_cache["model.layers.20.attn"]
    assert runner_kv_caches[1] is kv_cache["model.layers.28.attn"]


def test_bind_kv_cache_draft_model(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    layer_names = [
        "model.layers.0.attn",
        "model.layers.1.attn",
        "draft_model.layers.0.attn",
        "draft_model.layers.1.attn",
    ]
    ctx = {
        layer_name: Attention(32, 128, 0.1, prefix=layer_name)
        for layer_name in layer_names
    }
    kv_cache = {layer_name: torch.zeros((1,)) for layer_name in layer_names}
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["model.layers.0.attn"].kv_cache is kv_cache["model.layers.0.attn"]
    assert ctx["model.layers.1.attn"].kv_cache is kv_cache["model.layers.1.attn"]
    assert (
        ctx["draft_model.layers.0.attn"].kv_cache
        is kv_cache["draft_model.layers.0.attn"]
    )
    assert (
        ctx["draft_model.layers.1.attn"].kv_cache
        is kv_cache["draft_model.layers.1.attn"]
    )

    # caches are ordered by layer_index, interleaving target and draft model
    assert runner_kv_caches[0] is kv_cache["model.layers.0.attn"]
    assert runner_kv_caches[1] is kv_cache["draft_model.layers.0.attn"]
    assert runner_kv_caches[2] is kv_cache["model.layers.1.attn"]
    assert runner_kv_caches[3] is kv_cache["draft_model.layers.1.attn"]


def _memory_snapshot(total_gib: int, free_gib: int) -> MemorySnapshot:
    return MemorySnapshot(
        free_memory=free_gib * GiB_bytes,
        total_memory=total_gib * GiB_bytes,
        device="cpu",
        auto_measure=False,
    )


def test_request_memory_charges_external_weights():
    """Externally held weights are charged against the utilization budget;
    the engine is granted only the remainder."""
    cache_config = CacheConfig(gpu_memory_utilization=0.9)
    # 70 GiB of a 100 GiB device is held externally before the worker starts.
    snapshot = _memory_snapshot(total_gib=100, free_gib=30)

    with pytest.raises(ValueError, match="less than desired"):
        request_memory(snapshot, cache_config)

    # 90 GiB budget - 70 GiB external = 20 GiB for the engine.
    assert request_memory(snapshot, cache_config, 70 * GiB_bytes) == 20 * GiB_bytes

    # The external weights alone exceed the utilization budget.
    with pytest.raises(ValueError, match="exceed the desired"):
        request_memory(snapshot, cache_config, 95 * GiB_bytes)

    # Other tenants squeeze free memory below the engine's remainder.
    with pytest.raises(ValueError, match="less than the engine's budget"):
        request_memory(
            _memory_snapshot(total_gib=100, free_gib=10), cache_config, 70 * GiB_bytes
        )
