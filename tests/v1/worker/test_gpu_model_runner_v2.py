# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from tests.utils import spawn_new_process_for_each_test
from vllm.config import CacheConfig, LoRAConfig, ParallelConfig
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.model_runner import ExecuteModelState, GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.standalone_ar import speculator as ar_module
from vllm.v1.worker.gpu.spec_decode.utils import get_drafter_hidden_states


def test_non_last_pp_rank_uses_global_batch_for_sample_feedback(monkeypatch):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.is_last_pp_rank = False
    local_batch = object()
    global_batch = SimpleNamespace(idx_mapping=object())
    # Keep handles on the mocks: the runner's attributes are typed as the real
    # PCPManager / PPHandler / ModelState, which have no assert_* methods.
    restore_for_sampling = Mock()
    runner.pcp_manager = SimpleNamespace(
        global_batch=global_batch,
        restore_for_sampling=restore_for_sampling,
    )
    pp_receive = Mock(return_value=False)
    runner.pp_handler = SimpleNamespace(receive=pp_receive)
    postprocess_num_computed_tokens = Mock()
    monkeypatch.setattr(
        runner, "postprocess_num_computed_tokens", postprocess_num_computed_tokens
    )
    postprocess_state = Mock()
    runner.model_state = SimpleNamespace(postprocess_state=postprocess_state)
    runner.kv_connector = SimpleNamespace(post_forward=Mock(return_value=None))
    runner.eplb = SimpleNamespace(step=Mock())
    runner.execute_model_state = ExecuteModelState(
        input_batch=local_batch,
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=None,
        aux_hidden_states=None,
        dp_sync_state=None,
        finished_req_ids=set(),
        ec_connector_output=None,
        cudagraph_stats=None,
        num_spec_tokens_to_schedule=0,
    )

    runner.sample_tokens(None)

    pp_receive.assert_called_once_with(global_batch)
    postprocess_num_computed_tokens.assert_called_once_with(global_batch)
    postprocess_state.assert_called_once_with(global_batch.idx_mapping, 0)
    restore_for_sampling.assert_not_called()


def test_qsa_circular_group_uses_custom_slot_mapping(monkeypatch):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.max_model_len = 262144
    runner.is_encoder_decoder = False
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.cp_interleave = 1
    runner.cache_config = SimpleNamespace(enable_prefix_caching=True)
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
    )
    runner.parallel_config = parallel_config
    runner.vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
    )
    runner.jit_warmup_registry = JitWarmupRegistry(runner.vllm_config)
    runner.model_state = SimpleNamespace(
        get_additional_cg_support=lambda: (),
        num_new_sampled_tokens_per_step=1,
    )
    runner.speculator = None
    runner.req_states = []
    runner.input_buffers = SimpleNamespace(query_start_loc=None)
    runner.vocab_size = 1
    runner.max_num_reqs = 1
    runner.max_num_tokens = 2
    runner.device = torch.device("cuda")

    raw_spec = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    compressed_spec = FullAttentionSpec(
        block_size=262144,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["raw"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=8,
                    kv_cache_specs={"raw": raw_spec},
                ),
            ),
            KVCacheGroupSpec(layer_names=["compressed"], kv_cache_spec=compressed_spec),
        ],
    )

    class FakeAttnCGSupport:
        def narrow(self, *args):
            return self

    attn_cg_support = FakeAttnCGSupport()
    monkeypatch.setattr(
        model_runner_module,
        "init_attn_backend",
        lambda *args, **kwargs: ([], attn_cg_support, [8, 262144]),
    )
    monkeypatch.setattr(
        model_runner_module,
        "maybe_create_adaptive_verification_manager",
        lambda **kwargs: None,
    )

    captured = {}

    class BlockTablesCaptured(Exception):
        pass

    def capture_block_tables(**kwargs):
        captured.update(kwargs)
        raise BlockTablesCaptured

    monkeypatch.setattr(model_runner_module, "BlockTables", capture_block_tables)

    with pytest.raises(BlockTablesCaptured):
        runner.initialize_kv_cache(kv_cache_config)

    assert captured["max_num_blocks_per_group"] == [1, 1]
    assert captured["slot_mapping_enabled"] == [False, True]


@pytest.mark.parametrize(
    ("mamba_cache_mode", "num_speculative_blocks", "expected"),
    [
        pytest.param("align", 0, 65_536, id="align-prefix-cache"),
        pytest.param("none", 7, 8, id="no-prefix-cache-with-speculation"),
    ],
)
def test_initialize_kv_cache_does_not_dcp_shard_mamba_block_table(
    monkeypatch,
    mamba_cache_mode: str,
    num_speculative_blocks: int,
    expected: int,
):
    """Mamba/GDN block-table rows index global positions, unlike DCP KV."""
    max_model_len = 1_048_576
    attention_block_size = 1_536
    mamba_block_size = 16
    dcp_size = 8
    full_attention_spec = FullAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
        block_size=mamba_block_size,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], full_attention_spec),
            KVCacheGroupSpec(["kda"], mamba_spec),
        ],
    )
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=1,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
    )
    runner = SimpleNamespace(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        vllm_config=vllm_config,
        parallel_config=parallel_config,
    )

    class _CapturedWidths(Exception):
        pass

    captured: list[int] = []

    def capture_width(max_num_blocks: int, *_args, **_kwargs) -> int:
        captured.append(max_num_blocks)
        if len(captured) == 2:
            raise _CapturedWidths
        return max_num_blocks

    monkeypatch.setattr(model_runner_module, "get_block_table_width", capture_width)

    with pytest.raises(_CapturedWidths):
        GPUModelRunner.initialize_kv_cache(runner, kv_cache_config)

    # Attention KV is local to one of eight DCP ranks; KDA state is replicated
    # and therefore needs one table entry for every global 16-token page.
    assert captured == [86, expected]


def test_append_block_ids_rejects_write_past_row_capacity():
    """Reject an oversized staged write before it can corrupt the next row."""

    class _BlockTable:
        gpu = torch.empty((2, 4), dtype=torch.int32)

        def stage_write(self, *_args):
            pytest.fail("an oversized write must not be staged")

    block_tables = BlockTables.__new__(BlockTables)
    block_tables.num_kv_cache_groups = 1
    # append_block_ids only touches .gpu and .stage_write on each table.
    block_tables.block_tables = [_BlockTable()]  # type: ignore[list-item]  # duck-typed stub
    block_tables.num_blocks = SimpleNamespace(
        np=torch.tensor([[0, 3]], dtype=torch.int32)
    )

    with pytest.raises(
        RuntimeError,
        match=r"request 1, group 0 exceeds row capacity \(5 > 4\)",
    ):
        block_tables.append_block_ids(
            req_index=1,
            new_block_ids=([4, 5],),
            overwrite=False,
        )

    assert block_tables.num_blocks.np[0, 1] == 3


@contextlib.contextmanager
def _no_dummy_loras(
    lora_config: LoRAConfig | None, remove_lora: bool = True
) -> Iterator[None]:
    """No-op stand-in for LoRAModelRunnerMixin.maybe_setup_dummy_loras."""
    yield


def _make_capture_runner(
    monkeypatch: pytest.MonkeyPatch, captured: bool
) -> GPUModelRunner:
    """Minimal V2 runner for capture_model: fakes everything except the
    cudagraph_manager's needs_capture decision."""
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model_state = SimpleNamespace(
        supports_mm_inputs=False, capture_inner_cudagraphs=lambda *args: None
    )
    runner.cudagraph_manager = SimpleNamespace(
        needs_capture=lambda: captured,
        capture=lambda *args, **kwargs: None,
    )
    runner.lora_config = None
    monkeypatch.setattr(runner, "maybe_setup_dummy_loras", _no_dummy_loras)
    runner.speculator = None
    runner.adaptive_verification = None
    runner.model = None
    runner.input_buffers = None
    runner.pcp_manager = None
    runner.intermediate_tensors = None
    runner.block_tables = None
    runner.attn_groups = None
    runner.kv_cache_config = None
    runner.max_num_tokens = 2
    runner.use_aux_hidden_state_outputs = False
    runner.kv_connector = model_runner_module.NO_OP_KV_CONNECTOR
    return runner


def test_capture_model_locks_workspace_after_capture(monkeypatch):
    """A workspace resize after capture frees the buffer the captured graphs
    baked in, so capture_model must lock the workspace before returning
    (https://github.com/vllm-project/vllm/issues/55336)."""
    runner = _make_capture_runner(monkeypatch, captured=True)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    runner.capture_model()

    assert lock_calls == ["lock"]


def test_capture_model_skips_lock_when_nothing_captured(monkeypatch):
    """With no graphs to capture (e.g. enforce_eager) there is nothing baked
    into the workspace, so the early return must not lock it."""
    runner = _make_capture_runner(monkeypatch, captured=False)
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    assert runner.capture_model() == 0
    assert lock_calls == []


def test_capture_model_profile_only_skips_lock(monkeypatch):
    """The memory-profiling capture pass runs before kernel warmup and the
    real capture; locking there would stop the warmup from growing the
    workspace to its scheduler-realistic size."""
    runner = _make_capture_runner(monkeypatch, captured=True)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    runner.capture_model(profile_only=True)

    assert lock_calls == []


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="HiSparse requires NVIDIA CUDA"
)
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("nested_warmup", [False, True])
@spawn_new_process_for_each_test
def test_capture_model_does_not_consume_hisparse_mirror_phase(
    monkeypatch, tmp_path, vllm_runner, fail, nested_warmup
):
    """Real capture/dummy runs leave one live prefill mirror phase available."""
    import json

    from transformers import DeepseekV3Config

    from vllm.config import (
        AttentionConfig,
        CUDAGraphMode,
        HiSparseConfig,
        KVTransferConfig,
    )
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
        HiSparseConnector,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
        HiSparseConnectorWorker,
    )
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.core_client import InprocClient
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
    from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

    capability = current_platform.get_device_capability()
    if capability is None or capability.major < 9:
        pytest.skip("Sparse MLA requires Hopper or newer")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_DEEP_GEMM_WARMUP", "skip")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")

    # Same small MLA shape as the HiSparse E2E suite, with a local config and
    # dummy weights: no tokenizer, checkpoint or remote model is required.
    config = DeepseekV3Config(
        architectures=["DeepseekV32ForCausalLM"],
        num_hidden_layers=8,
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=8,
        num_key_value_heads=1,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=3,
        moe_intermediate_size=2048,
        n_group=8,
        topk_group=4,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=128,
        vocab_size=128,
        max_position_embeddings=163840,
        rope_scaling={
            "type": "yarn",
            "factor": 40,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
        bos_token_id=0,
        eos_token_id=1,
        torch_dtype="bfloat16",
    ).to_dict()
    config["model_type"] = "deepseek_v32"
    (tmp_path / "config.json").write_text(json.dumps(config))

    retained_graph_owners: list[object] = []
    with (
        contextlib.ExitStack() as graph_lifetime,
        vllm_runner(
            str(tmp_path),
            load_format="dummy",
            skip_tokenizer_init=True,
            dtype="bfloat16",
            seed=7,
            max_model_len=64,
            max_num_batched_tokens=8,
            max_num_seqs=1,
            block_size=64,
            num_gpu_blocks_override=128,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            async_scheduling=False,
            attention_config=AttentionConfig(
                hisparse_config=HiSparseConfig(device_buffer_size=512)
            ),
            kv_transfer_config=KVTransferConfig(
                kv_connector="HiSparseConnector",
                kv_role="kv_both",
                kv_connector_extra_config={"host_pool_gib": 1 / 64},
            ),
            compilation_config={
                "mode": 0,
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "cudagraph_capture_sizes": [1, 2],
            },
        ) as engine,
    ):
        client = engine.llm.llm_engine.engine_core
        assert isinstance(client, InprocClient)
        runner = client.engine_core.model_executor.driver_worker.worker.model_runner
        assert isinstance(runner, GPUModelRunner)
        connector = runner.kv_connector
        assert isinstance(connector, ActiveKVConnector)
        assert isinstance(connector.kv_connector, HiSparseConnector)
        connector_worker = connector.kv_connector.connector_worker
        assert isinstance(connector_worker, HiSparseConnectorWorker)
        worker: HiSparseConnectorWorker = connector_worker
        assert worker.cache_handles
        cudagraph_manager = runner.cudagraph_manager
        assert isinstance(cudagraph_manager, ModelCudaGraphManager)
        manager: ModelCudaGraphManager = cudagraph_manager
        # ExitStack closes after the engine, including when an assertion fails.
        graph_lifetime.callback(retained_graph_owners.clear)
        graph_lifetime.callback(torch.accelerator.synchronize)

        def release_graphs_with_live_pool_owners():
            # Reuse the real pool without destroying its last captured owner.
            # Breakable captures own their CUDAGraph segments via bound replay.
            retained_graph_owners.extend(manager.graphs.values())
            breakable_runner = manager.breakable_cg_runner
            assert breakable_runner is not None
            retained_graph_owners.extend(
                entry.capture
                for entry in breakable_runner.entries.values()
                if entry.capture is not None
            )
            manager.release_graphs()

        assert manager.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
        assert manager.use_breakable_cg
        assert manager.captured_token_counts()
        torch.accelerator.synchronize()
        host_before = [
            cache.runtime.host_cache.view(torch.uint8).clone()
            for cache in worker.cache_handles
        ]
        completion_before = worker.host_write_event

        def assert_mirror_phase_unused():
            torch.accelerator.synchronize()
            breakable_runner = manager.breakable_cg_runner
            assert breakable_runner is not None
            assert breakable_runner.entries
            assert all(
                entry.capture is not None for entry in breakable_runner.entries.values()
            )
            assert not worker._per_layer_mirrored
            assert not worker._submitted_mirror_layers
            assert worker.host_write_event is completion_before
            for cache, expected in zip(worker.cache_handles, host_before):
                assert torch.equal(cache.runtime.host_cache.view(torch.uint8), expected)

        # Public release removes startup graphs from dispatch. Keep their real
        # owners alive locally while recapturing into the same Torch pool.
        release_graphs_with_live_pool_owners()
        outer = connector.disabled() if nested_warmup else contextlib.nullcontext()
        with outer:
            if nested_warmup:
                runner._dummy_run(2, skip_eplb=True)
                assert not has_kv_transfer_group()
            runner.capture_model(profile_only=True)
            assert manager.captured_token_counts()
            assert has_kv_transfer_group() is not nested_warmup
            if fail:
                with pytest.raises(AssertionError, match="Graph already captured"):
                    runner.capture_model(profile_only=True)
                assert has_kv_transfer_group() is not nested_warmup
            if nested_warmup:
                runner._dummy_run(2, skip_eplb=True)
                assert not has_kv_transfer_group()
            assert_mirror_phase_unused()
        assert get_kv_transfer_group() is connector.kv_connector
        if fail:
            release_graphs_with_live_pool_owners()
            runner.capture_model(profile_only=True)
            assert manager.captured_token_counts()
        assert_mirror_phase_unused()
        assert get_kv_transfer_group() is connector.kv_connector

        # Three prompt tokens exceed the PW capture sizes, so this is a real
        # non-decode prefill, rather than a replay or a synthetic handle call.
        class MirrorPhaseObservation:
            observed = False
            per_layer_mirrored: set[int] | None = None
            submitted_mirror_layers: set[int] | None = None
            completion_event: torch.Event | None = None
            rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None

            def __call__(self, module, inputs, model_output):
                available = is_forward_context_available()
                context = get_forward_context() if available else None
                details = {
                    "forward_context_available": available,
                    "mode": None if context is None else context.cudagraph_runtime_mode,
                    "mirrored": sorted(worker._per_layer_mirrored),
                    "submitted": sorted(worker._submitted_mirror_layers),
                    "active_tokens": [
                        c.num_actual_tokens for c in worker.cache_handles
                    ],
                    "decode_batch": [c.decode_batch for c in worker.cache_handles],
                }
                assert not self.observed, details
                assert available, details
                assert context is not None, details
                assert context.cudagraph_runtime_mode == CUDAGraphMode.NONE, details
                self.observed = True
                self.per_layer_mirrored = set(worker._per_layer_mirrored)
                self.submitted_mirror_layers = set(worker._submitted_mirror_layers)
                # This model uses finish_forward's supported whole-step mirror.
                # QSA reference cases exercise the real per-layer duplicate guard.
                assert worker._per_layer_mirrored == set(), details
                assert worker._submitted_mirror_layers == set(), details
                self.completion_event = worker.host_write_event
                self.rows = []
                for cache in worker.cache_handles:
                    assert not cache.decode_batch and cache.num_actual_tokens == 3, (
                        details
                    )
                    slot_mapping = cache.slot_mapping
                    mirror_slot_mapping = cache.mirror_slot_mapping
                    cache_view = cache.view
                    assert slot_mapping is not None, details
                    assert mirror_slot_mapping is not None, details
                    assert cache_view is not None, details
                    slots = slot_mapping[:3].clone()
                    destinations = mirror_slot_mapping[:3].clone()
                    rows = cache_view.cache[
                        slots // worker.kernel_block_size,
                        slots % worker.kernel_block_size,
                    ].clone()
                    self.rows.append((slots, destinations, rows))
                # A public forward hook returning None preserves model_output.
                return None

        phase = MirrorPhaseObservation()
        hook = runner.get_model().register_forward_hook(phase)
        try:
            outputs = engine.llm.generate(
                {"prompt_token_ids": [2, 3, 4]},
                SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
                use_tqdm=False,
            )
        finally:
            hook.remove()
        assert phase.observed
        assert phase.per_layer_mirrored == set()
        assert phase.submitted_mirror_layers == set()
        assert len(outputs) == 1 and len(outputs[0].outputs[0].token_ids) == 1
        assert phase.completion_event is not None
        phase.completion_event.synchronize()
        assert phase.completion_event is not completion_before
        assert worker.host_write_event is not completion_before
        assert phase.rows is not None and len(phase.rows) == len(worker.cache_handles)
        nonzero = False
        for cache, (slots, destinations, rows) in zip(worker.cache_handles, phase.rows):
            destinations = destinations.cpu()
            assert torch.all(slots >= 0) and torch.all(destinations >= 0)
            rows = rows.cpu()
            actual = cache.runtime.host_cache[destinations]
            torch.testing.assert_close(
                actual.view(torch.uint8), rows.view(torch.uint8), rtol=0, atol=0
            )
            nonzero |= bool(torch.count_nonzero(rows).item())
        assert nonzero


@pytest.mark.parametrize("target_buffer", ["absent", "none", "tensor"])
def test_get_drafter_hidden_states_tolerates_missing_target_buffer(target_buffer):
    """Targets allocate the MTP hidden buffer only for hidden-state drafters."""
    hidden_states = torch.zeros(4, 8)
    buffer = torch.arange(16 * 8, dtype=torch.float32).view(16, 8)
    if target_buffer == "absent":
        model = SimpleNamespace()
    else:
        returned = buffer if target_buffer == "tensor" else None
        model = SimpleNamespace(get_mtp_target_hidden_states=lambda: returned)

    out = get_drafter_hidden_states(model, hidden_states)

    if target_buffer == "tensor":
        assert torch.equal(out, buffer[:4])
    else:
        assert out is hidden_states


def test_async_copy_to_np_does_not_alias_reused_buffer():
    buffer = torch.zeros(4, dtype=torch.int64)

    snapshot = async_copy_to_np(buffer)
    buffer.fill_(1)

    assert snapshot.tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize(
    ("draft_dtype", "expected_dtype"),
    [("auto", "auto"), ("bfloat16", "bfloat16"), (None, "fp8_e4m3")],
)
def test_standalone_draft_kv_dtype_override(monkeypatch, draft_dtype, expected_dtype):
    @dataclass
    class Config:
        cache_config: CacheConfig
        parallel_config: ParallelConfig
        model_config: object
        quant_config: object | None = None

    target = Config(CacheConfig(cache_dtype="fp8_e4m3"), ParallelConfig(), object())
    speculator = object.__new__(ar_module.StandaloneARSpeculator)
    speculator.vllm_config = target
    speculator.draft_model_config = object()
    speculator.speculative_config = SimpleNamespace(
        kv_cache_dtype=draft_dtype, draft_parallel_config=ParallelConfig()
    )
    get_model = Mock(return_value=object())
    monkeypatch.setattr(ar_module, "get_model", get_model)
    monkeypatch.setattr(ar_module, "set_model_tag", lambda _: contextlib.nullcontext())

    speculator.load_draft_model(object(), set())

    draft_config = get_model.call_args.kwargs["vllm_config"]
    assert draft_config.cache_config.cache_dtype == expected_dtype
    assert speculator.attn_vllm_config.cache_config.cache_dtype == expected_dtype
    assert target.cache_config.cache_dtype == "fp8_e4m3"
