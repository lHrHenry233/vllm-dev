# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from dataclasses import fields, replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tests.v1.attention.test_gdn_metadata_builder import (
    BLOCK_SIZE,
    DEVICE,
    _create_gdn_builder,
)
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.worker.gpu.model_states import mamba_hybrid
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState
from vllm.v1.worker.utils import AttentionGroup


def _recoverssm_attn_groups(layer_name: str) -> list[list[AttentionGroup]]:
    """RecoverSSMState.record_step reads only each group's ``layer_names``."""
    group = SimpleNamespace(layer_names=[layer_name])
    return [[group]]  # type: ignore[list-item]  # layer_names-only stand-in


@pytest.mark.parametrize("capture", [False, True])
def test_gdn_metadata_accepts_live_slot_mapping_rows(capture, tmp_path):
    """Tuple rows preserve the real builder's Tensor-input metadata contract."""
    from transformers import Qwen3Config

    from tests.v1.attention.utils import create_vllm_config
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
    from vllm.v1.worker.gpu.attn_utils import build_attn_metadata

    Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        vocab_size=32,
    ).save_pretrained(tmp_path)
    vllm_config = create_vllm_config(
        model_name=str(tmp_path), max_num_seqs=2, max_num_batched_tokens=16
    )
    vllm_config.compilation_config.cudagraph_mode = (
        CUDAGraphMode.FULL if capture else CUDAGraphMode.NONE
    )
    builder = GDNAttentionMetadataBuilder(
        MambaSpec(block_size=BLOCK_SIZE, shapes=((16, 64),), dtypes=(torch.float16,)),
        ["layer.0"],
        vllm_config,
        DEVICE,
    )
    group = AttentionGroup(
        MambaAttentionBackendEnum.GDN_ATTN.get_class(),
        ["layer.0"],
        builder.kv_cache_spec,
        0,
        metadata_builders=[builder],
    )
    config = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(group.layer_names, group.kv_cache_spec)],
    )
    common = create_common_attn_metadata(
        BatchSpec([40], [1]), BLOCK_SIZE, DEVICE, arange_block_indices=True
    )
    slots = common.slot_mapping.unsqueeze(0)
    args = dict(
        attn_groups=[[group]],
        num_reqs=1,
        num_tokens=1,
        query_start_loc_gpu=common.query_start_loc,
        query_start_loc_cpu=common.query_start_loc_cpu,
        max_query_len=1,
        seq_lens=common.seq_lens,
        max_seq_len=40,
        block_tables=[common.block_table_tensor],
        kv_cache_config=config,
        seq_lens_cpu_upper_bound=common.seq_lens_cpu_upper_bound,
        for_cudagraph_capture=capture,
    )
    reference = build_attn_metadata(slot_mappings=slots, **args)["layer.0"]
    expected = reference.non_spec_state_indices_tensor.clone()
    rows = slots.unbind(0)
    slots.add_(11)
    actual = build_attn_metadata(slot_mappings=rows, **args)["layer.0"]
    torch.testing.assert_close(actual.non_spec_state_indices_tensor, expected)
    assert actual.num_actual_tokens == reference.num_actual_tokens


@contextmanager
def _real_mamba_runtime(tmp_path, num_speculative_tokens, num_groups=2):
    """Bind tiny real GDN layers without loading a model checkpoint."""
    from transformers import Qwen3NextConfig

    from tests.v1.attention.utils import create_vllm_config
    from vllm.config import SpeculativeConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationMode
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.gpu.states import RequestState

    device = torch.device("cuda")
    Qwen3NextConfig(
        architectures=["Qwen3NextForCausalLM"],
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_groups,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * num_groups,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        decoder_sparse_step=1,
        vocab_size=32,
        max_position_embeddings=128,
    ).save_pretrained(tmp_path)
    config = create_vllm_config(
        model_name=str(tmp_path),
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=16,
        dtype=torch.float16,
    )
    config.cache_config.mamba_cache_mode = "align"
    config.cache_config.mamba_block_size = BLOCK_SIZE
    config.compilation_config.mode = CompilationMode.NONE
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    if num_speculative_tokens:
        config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    with set_current_vllm_config(config):
        with torch.device(device):
            model = Qwen3NextForCausalLM(vllm_config=config)
        layers = [
            (name, layer)
            for name, layer in config.compilation_config.static_forward_context.items()
            if isinstance(layer, QwenGatedDeltaNetAttention)
        ]
        assert len(layers) == num_groups
        specs = []
        for _, layer in layers:
            spec = layer.get_kv_cache_spec(config)
            assert isinstance(spec, MambaSpec)
            layer.bind_kv_cache(
                torch.zeros(
                    (256, 1, 1, spec.page_size_bytes),
                    dtype=torch.int8,
                    device=device,
                )
            )
            specs.append(spec)
        kv_config = KVCacheConfig(
            num_blocks=256,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec([name], spec) for (name, _), spec in zip(layers, specs)
            ],
        )
        groups = []
        for i, ((name, layer), spec) in enumerate(zip(layers, specs)):
            group = AttentionGroup(layer.get_attn_backend(), [name], spec, i)
            group.create_metadata_builders(config, device)
            groups.append([group])
        tables = BlockTables([BLOCK_SIZE] * num_groups, 4, 16, [8] * num_groups, device)
        state = MambaHybridModelState(config, model, None, device)
        state.initialize_kv_cache(kv_config, tables)
        requests = RequestState(4, 128, 16, num_speculative_tokens, 32, device)
        for row, req_id in enumerate(("a", "b")):
            tokens = [0] * 39
            requests.add_request(req_id, 39, tokens, 39, 64)
            block_ids = tuple(
                list(range(10 + 50 * group + 8 * row, 18 + 50 * group + 8 * row))
                for group in range(num_groups)
            )
            tables.append_block_ids(
                requests.req_id_to_index[req_id], block_ids, overwrite=True
            )
            state.add_request(
                requests.req_id_to_index[req_id],
                NewRequestData(
                    req_id=req_id,
                    prompt_token_ids=tokens,
                    mm_features=[],
                    sampling_params=None,
                    pooling_params=None,
                    block_ids=block_ids,
                    num_computed_tokens=39,
                    lora_request=None,
                    prefill_token_ids=tokens,
                ),
            )
        requests.apply_staged_writes()
        tables.apply_staged_writes()
        buffers = InputBuffers(4, 16, device)
        yield state, tables, requests, buffers, groups, kv_config


def _write_group_blocks(tables, requests, bases):
    for row, req_id in enumerate(("a", "b")):
        tables.append_block_ids(
            requests.req_id_to_index[req_id],
            tuple(list(range(base + 8 * row, base + 8 * row + 8)) for base in bases),
            overwrite=True,
        )
    tables.apply_staged_writes()


def _prepare_real_mamba_metadata(runtime, mode, capture, active=2, padded=4):
    from vllm.v1.worker.gpu.input_batch import InputBatch

    state, tables, requests, buffers, groups, kv_config = runtime
    padded = padded if mode == CUDAGraphMode.FULL else active
    batch = InputBatch.make_dummy(
        active, active, buffers, max_query_len=1, is_padding=False
    )
    req_ids = ["a", "b"][:active]
    mapping = np.array([requests.req_id_to_index[r] for r in req_ids], dtype=np.intp)
    qsl = np.array(
        [0, *range(1, active + 1), *([active] * (padded - active))], dtype=np.int32
    )
    buffers.seq_lens[:active].fill_(40)
    buffers.positions[:active].fill_(39)
    buffers.query_start_loc[: padded + 1].copy_(torch.from_numpy(qsl))
    idx_mapping = torch.tensor(mapping, device=state.device)
    batch = replace(
        batch,
        req_ids=req_ids,
        idx_mapping=idx_mapping,
        expanded_idx_mapping=idx_mapping,
        idx_mapping_np=mapping,
        num_computed_tokens_np=requests.num_computed_tokens_np[mapping].copy(),
        num_computed_prefill_tokens_np=requests.num_computed_prefill_tokens[
            mapping
        ].copy(),
        prefill_len_np=requests.prefill_len.np[mapping].copy(),
        num_reqs_after_padding=padded,
        num_tokens_after_padding=padded,
        query_start_loc=buffers.query_start_loc[: padded + 1],
        query_start_loc_np=qsl,
        seq_lens=buffers.seq_lens[:padded],
        seq_lens_cpu_upper_bound=torch.tensor([40] * active + [0] * (padded - active)),
        positions=buffers.positions[:padded],
    )
    block_tables = tables.gather_block_tables(batch.idx_mapping, padded)
    slots = tables.compute_slot_mappings(
        batch.idx_mapping,
        batch.query_start_loc,
        batch.positions,
        padded,
    )
    metadata = state.prepare_attn(
        batch,
        mode,
        block_tables,
        slots,
        groups,
        kv_config,
        for_capture=capture,
    )
    return metadata, slots


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_shared_gdn_batch_metadata_is_prepared_once(tmp_path, monkeypatch, dist_init):
    """FULL metadata shares batch buffers while real groups retain distinct IDs."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with _real_mamba_runtime(tmp_path, 0, num_groups=3) as runtime:
        _write_group_blocks(runtime[1], runtime[2], (10, 60, 110))
        actual, _ = _prepare_real_mamba_metadata(runtime, CUDAGraphMode.FULL, False)
        values = list(actual.values())
        assert [value.non_spec_state_indices_tensor[0].item() for value in values] == [
            12,
            62,
            112,
        ]
        assert values[0].non_spec_query_start_loc is not None
        group_fields = {
            "spec_state_indices_tensor",
            "non_spec_state_indices_tensor",
            "prefill_state_indices",
            "checkpoint",
        }
        for value in values[1:]:
            for field in fields(value):
                if field.name not in group_fields:
                    assert getattr(value, field.name) is getattr(values[0], field.name)


def test_prepare_attn_forwards_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=0)
    state.model_config = SimpleNamespace(max_model_len=8192)
    state._align_mode = False
    state.recoverssm = None

    positions = torch.tensor([1536], dtype=torch.int64)
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=1,
        num_reqs_after_padding=1,
        num_tokens_after_padding=1,
        query_start_loc_np=torch.tensor([0, 1], dtype=torch.int32).numpy(),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        num_scheduled_tokens=torch.tensor([1], dtype=torch.int32),
        max_query_len=None,
        seq_lens_cpu_upper_bound=torch.tensor([1537], dtype=torch.int32),
        seq_lens=torch.tensor([1537], dtype=torch.int32),
        is_prefilling_np=torch.tensor([False]).numpy(),
        prefill_runs_as_decode_np=None,
        dcp_local_seq_lens=None,
        positions=positions,
        prompt_lens=torch.tensor([1024], dtype=torch.int32),
    )
    expected_metadata = {"layer": object()}
    build_attn_metadata = Mock(return_value=expected_metadata)
    monkeypatch.setattr(mamba_hybrid, "build_attn_metadata", build_attn_metadata)

    metadata = state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=torch.empty(0, dtype=torch.int64),
        attn_groups=[],
        kv_cache_config=Mock(),
    )

    assert metadata is expected_metadata
    assert build_attn_metadata.call_args.kwargs["positions"] is positions


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("num_speculative_tokens", [0, 2])
@pytest.mark.parametrize(
    "mode,capture",
    [
        (CUDAGraphMode.NONE, False),
        (CUDAGraphMode.FULL, False),
        (CUDAGraphMode.FULL, True),
    ],
)
def test_prepare_attn_keeps_live_aligned_group_views(
    tmp_path,
    monkeypatch,
    dist_init,
    num_speculative_tokens,
    mode,
    capture,
):
    """Real gathered views survive updates; public cache rebind needs recapture."""
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
    from vllm.v1.worker.gpu.block_table import BlockTables

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with _real_mamba_runtime(tmp_path, num_speculative_tokens) as runtime:
        state, tables, requests, buffers, groups, kv_config = runtime
        _write_group_blocks(tables, requests, (10, 60))

        def prepare(active=2, padded=4):
            metadata, slots = _prepare_real_mamba_metadata(
                runtime, mode, capture, active, padded
            )
            views = tuple(
                g[0].get_metadata_builder(0).mamba_aligned_state_indices for g in groups
            )
            assert all(view is not None for view in views)
            return metadata, views, slots

        def indices(metadata):
            return [
                m.spec_state_indices_tensor
                if m.num_spec_decodes
                else m.non_spec_state_indices_tensor
                for m in metadata.values()
            ]

        def expected(active, padded, delta):
            rows = [[[12, 13, 14], [20, 21, 22]], [[62, 63, 64], [70, 71, 72]]]
            result = []
            for group in rows:
                live = [
                    [x + delta for x in row[: num_speculative_tokens + 1]]
                    for row in group[:active]
                ]
                live += [[NULL_BLOCK_ID] * (num_speculative_tokens + 1)] * (
                    padded - active
                )
                value = torch.tensor(live, dtype=torch.int32, device=state.device)
                result.append(value if num_speculative_tokens else value[:, 0])
            return result

        first, original_views, slots = prepare()
        for actual, reference in zip(
            indices(first), expected(2, 4 if mode == CUDAGraphMode.FULL else 2, 0)
        ):
            torch.testing.assert_close(actual, reference)
        graph, observed = None, None
        if capture:
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                observed = [value.clone() for value in indices(first)]

        _write_group_blocks(tables, requests, (30, 80))
        second, current_views, current_slots = prepare()
        assert all(now is old for now, old in zip(current_views, original_views))
        for actual, reference in zip(
            indices(second), expected(2, 4 if mode == CUDAGraphMode.FULL else 2, 20)
        ):
            torch.testing.assert_close(actual, reference)
        if graph is not None:
            graph.replay()
            for actual, reference in zip(observed, expected(2, 4, 20)):
                torch.testing.assert_close(actual, reference)
        assert tables.get_slot_mapping_rows(
            current_slots
        ) is tables.get_slot_mapping_rows(slots)
        expected_slots = [[519, 647], [1319, 1447]]
        if mode == CUDAGraphMode.FULL:
            expected_slots = [row + [-1, -1] for row in expected_slots]
        torch.testing.assert_close(
            current_slots,
            torch.tensor(expected_slots, dtype=torch.int64, device=state.device),
        )

        _write_group_blocks(tables, requests, (50, 100))
        third, views, _ = prepare(1)
        if mode == CUDAGraphMode.FULL:
            assert all(now is old for now, old in zip(views, original_views))
        else:
            assert all(now is not old for now, old in zip(views, original_views))
        for actual, reference in zip(
            indices(third), expected(1, 4 if mode == CUDAGraphMode.FULL else 1, 40)
        ):
            torch.testing.assert_close(actual, reference)
        if graph is not None:
            graph.replay()
            for actual, reference in zip(observed, expected(1, 4, 40)):
                torch.testing.assert_close(actual, reference)
        _, short_views, _ = prepare(1, 1)
        assert all(
            view.shape == (1, num_speculative_tokens + 1) for view in short_views
        )
        old_values = [view.clone() for view in short_views]

        tables = BlockTables([BLOCK_SIZE] * 2, 4, 16, [8] * 2, state.device)
        state.initialize_kv_cache(kv_config, tables)
        runtime = state, tables, requests, buffers, groups, kv_config
        _write_group_blocks(tables, requests, (70, 120))
        rebound, rebound_views, _ = prepare(1, 1)
        assert all(now is not old for now, old in zip(rebound_views, short_views))
        for old, reference in zip(short_views, old_values):
            torch.testing.assert_close(old, reference)
        for actual, reference in zip(indices(rebound), expected(1, 1, 60)):
            torch.testing.assert_close(actual, reference)
        if capture:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                observed = [value.clone() for value in indices(rebound)]
            graph.replay()
            for actual, reference in zip(observed, expected(1, 1, 60)):
                torch.testing.assert_close(actual, reference)

        group = groups[1][0]
        replacement = AttentionGroup(
            group.backend, group.layer_names, group.kv_cache_spec, 1
        )
        replacement.create_metadata_builders(state.vllm_config, state.device)
        groups[1] = [replacement]
        _write_group_blocks(tables, requests, (90, 140))
        changed, changed_views, _ = prepare(1, 1)
        assert changed_views[1] is rebound_views[1]
        for actual, reference in zip(indices(changed), expected(1, 1, 80)):
            torch.testing.assert_close(actual, reference)


def test_add_request_seeds_state_idx_in_mamba_blocks() -> None:
    """A drafter group can lower cache_config.block_size below the mamba
    block size; the seeded column must still be in mamba blocks."""
    state = object.__new__(MambaHybridModelState)
    state.cache_config = SimpleNamespace(block_size=16, mamba_block_size=880)
    state._align_mode = True
    state.rope_state = None
    state.prompt_embeds_state = None
    state.num_accepted_tokens_gpu = torch.ones(2, dtype=torch.int32)
    state._mamba_state_idx_gpu = torch.zeros(2, dtype=torch.int32)

    state.add_request(1, SimpleNamespace(num_computed_tokens=107_360))

    assert state._mamba_state_idx_gpu[1] == 121


def test_padded_prompt_tail_builds_as_spec_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-token prompt tail over prior state, padded with K placeholder
    drafts (e.g. a P/D decode-node arrival), must reach the GDN builder as a
    spec-decode row. Built as a prefill, the placeholder tokens are folded into
    the recurrent state and can't be rolled back.
    """
    k = 3
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=k)
    state.model_config = SimpleNamespace(max_model_len=8192)
    state._align_mode = False
    state.recoverssm = None
    state.num_accepted_tokens_gpu = torch.ones(4, dtype=torch.int32)

    # A verify decode, a padded prompt tail (128 of 129 prompt tokens already
    # computed), and a fresh prompt chunk of the same length.
    query_lens = [k + 1] * 3
    seq_lens = [50, 128 + k + 1, k + 1]
    is_prefilling = [False, True, True]
    query_start_loc = np.array([0, 4, 8, 12], dtype=np.int32)
    input_batch = SimpleNamespace(
        num_reqs=3,
        num_tokens=12,
        num_reqs_after_padding=3,
        num_tokens_after_padding=12,
        idx_mapping=torch.arange(3),
        query_start_loc_np=query_start_loc,
        query_start_loc=torch.from_numpy(query_start_loc),
        num_scheduled_tokens=np.array(query_lens, dtype=np.int32),
        num_draft_tokens_per_req=np.array([k, k, 0], dtype=np.int32),
        max_query_len=None,
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        is_prefilling_np=np.array(is_prefilling),
        prefill_runs_as_decode_np=np.array([False, True, False]),
        dcp_local_seq_lens=None,
        positions=torch.zeros(12, dtype=torch.int64),
        prompt_lens=None,
    )
    build_attn_metadata = Mock(return_value={})
    monkeypatch.setattr(mamba_hybrid, "build_attn_metadata", build_attn_metadata)
    state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=torch.empty(0, dtype=torch.int64),
        attn_groups=[],
        kv_cache_config=Mock(),
    )
    mamba_metadata = build_attn_metadata.call_args.kwargs[
        "model_specific_attn_metadata"
    ]
    assert mamba_metadata.is_prefilling.tolist() == [False, False, True]

    builder = _create_gdn_builder(num_speculative_tokens=k)
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens), BLOCK_SIZE, DEVICE
    ).replace(**mamba_metadata.get_extra_common_attn_kwargs(0, 3))
    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=mamba_metadata.num_accepted_tokens,
        num_decode_draft_tokens_cpu=mamba_metadata.num_decode_draft_tokens_cpu,
    )

    # Only the fresh prompt chunk needs the prefill kernels.
    assert meta.num_spec_decodes == 2
    assert meta.num_prefills == 1
    assert meta.num_prefill_tokens == k + 1


def _mamba_hybrid_state(num_speculative_tokens: int) -> MambaHybridModelState:
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
    state.model_config = SimpleNamespace(max_model_len=8192)
    state._align_mode = False
    state.recoverssm = None
    # A request that accepted 3 drafts (plus the bonus token) on the previous
    # step; the others have never had a draft accepted.
    state.num_accepted_tokens_gpu = torch.tensor([4, 1, 1], dtype=torch.int32)
    return state


def _input_batch(
    num_scheduled_tokens: list[int],
    num_draft_tokens_per_req: list[int] | None,
    is_prefilling: list[bool],
    num_reqs_after_padding: int | None = None,
) -> SimpleNamespace:
    """`num_draft_tokens_per_req=None` is a batch where nobody proposed a
    draft this step, e.g. because ngram found no match for anyone."""
    num_reqs = len(num_scheduled_tokens)
    query_start_loc = np.zeros(num_reqs + 1, dtype=np.int32)
    query_start_loc[1:] = np.cumsum(num_scheduled_tokens)
    num_tokens = int(query_start_loc[-1])
    seq_lens = torch.full((num_reqs,), 10, dtype=torch.int32)
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs_after_padding or num_reqs,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        query_start_loc_np=query_start_loc,
        num_scheduled_tokens=np.array(num_scheduled_tokens, dtype=np.int32),
        max_query_len=None,
        seq_lens_cpu_upper_bound=seq_lens,
        is_prefilling_np=np.array(is_prefilling),
        prefill_runs_as_decode_np=None,
        num_draft_tokens_per_req=None
        if num_draft_tokens_per_req is None
        else np.array(num_draft_tokens_per_req, dtype=np.int32),
        idx_mapping=torch.arange(num_reqs),
        query_start_loc=torch.from_numpy(query_start_loc),
        seq_lens=seq_lens,
        dcp_local_seq_lens=None,
        positions=torch.zeros(num_tokens, dtype=torch.int64),
        prompt_lens=None,
    )


def _prepare_attn_metadata(
    state: MambaHybridModelState,
    monkeypatch,
    input_batch: SimpleNamespace | None = None,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
) -> SimpleNamespace:
    captured = {}

    def fake_build_attn_metadata(**kwargs):
        captured.update(kwargs)
        return Mock()

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.model_states.mamba_hybrid.build_attn_metadata",
        fake_build_attn_metadata,
    )

    if input_batch is None:
        input_batch = _input_batch([1, 1], None, [False, False])
    state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=cudagraph_mode,
        block_tables=(),
        slot_mappings={},
        attn_groups=[],
        kv_cache_config=Mock(),
    )
    return captured["model_specific_attn_metadata"]


def test_prepare_attn_keeps_the_accepted_offset_when_the_batch_drafted_nothing(
    monkeypatch,
):
    """`num_accepted_tokens` must reach the attention builders whenever the
    speculative config is on, even when *this* batch scheduled no draft
    tokens at all. Before the fix, `use_spec_decode` was gated on this step's
    scheduled drafts (`len(scheduled_spec_decode_tokens) > 0` in the V1
    runner) rather than on the speculative config, so a request that accepted
    drafts on a previous step would silently lose that offset the moment
    ngram found no match for anyone in the batch.
    """
    state = _mamba_hybrid_state(num_speculative_tokens=3)

    metadata = _prepare_attn_metadata(state, monkeypatch)

    assert metadata.num_accepted_tokens is not None
    assert metadata.num_accepted_tokens.tolist() == [4, 1]
    assert metadata.num_decode_draft_tokens_cpu is not None


def test_prepare_attn_omits_the_offset_without_a_speculative_config(monkeypatch):
    """Mirrors the check above: with speculative decoding off entirely, there
    is no accepted-token offset to carry, and the metadata must say so
    explicitly (`None`) rather than a stale or default value."""
    state = _mamba_hybrid_state(num_speculative_tokens=0)

    metadata = _prepare_attn_metadata(state, monkeypatch)

    assert metadata.num_accepted_tokens is None
    assert metadata.num_decode_draft_tokens_cpu is None


def test_zero_draft_decode_row_is_marked_as_a_speculative_row(monkeypatch):
    """A decode row that got no drafts still has to reach the speculative path.

    The recurrent backends only apply a row's accepted-token offset on that
    path, and a row that accepted drafts on the previous step keeps that offset
    even when the current step proposes nothing (the scheduler drops drafts
    whenever the token budget truncates the step). Marking such a row with the
    -1 sentinel sends it to the plain decode path, which reads the state at
    column 0 and ignores the offset.
    """
    state = _mamba_hybrid_state(num_speculative_tokens=3)
    input_batch = _input_batch([3, 1, 1], [2, 0, 0], [False, False, True])

    metadata = _prepare_attn_metadata(state, monkeypatch, input_batch)

    assert metadata.num_decode_draft_tokens_cpu.tolist() == [
        2,  # decode row with drafts
        0,  # decode row with no drafts: speculative path, offset applied
        -1,  # one-token prefill tail: no offset to apply, stays a plain row
    ]


def test_batch_without_any_drafts_still_marks_its_decode_rows(monkeypatch):
    """`num_draft_tokens_per_req` is None whenever no row in the batch drafted
    anything. Those rows are ordinary decode rows carrying an offset from the
    previous step, so the absence of drafts in this step must not be read as
    speculative decoding being off."""
    state = _mamba_hybrid_state(num_speculative_tokens=3)
    input_batch = _input_batch([1, 1], None, [False, True])

    metadata = _prepare_attn_metadata(state, monkeypatch, input_batch)

    assert metadata.num_decode_draft_tokens_cpu.tolist() == [0, -1]


def test_padded_rows_keep_the_sentinel(monkeypatch):
    """Full CUDA-graph capture pads the batch; padded rows describe no request
    and must not be picked up as speculative rows."""
    state = _mamba_hybrid_state(num_speculative_tokens=3)
    input_batch = _input_batch([1, 1], [0, 0], [False, False], num_reqs_after_padding=4)

    metadata = _prepare_attn_metadata(
        state, monkeypatch, input_batch, cudagraph_mode=CUDAGraphMode.FULL
    )

    assert metadata.num_decode_draft_tokens_cpu.tolist() == [0, 0, -1, -1]


def test_rows_wider_than_the_speculative_state_window_keep_the_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _mamba_hybrid_state(num_speculative_tokens=7)
    input_batch = _input_batch([12, 8, 1], None, [False, False, False])

    metadata = _prepare_attn_metadata(state, monkeypatch, input_batch)

    assert metadata.num_decode_draft_tokens_cpu.tolist() == [-1, 0, 0]


def test_chunked_prefill_tail_of_two_or_three_tokens_keeps_the_sentinel(
    monkeypatch,
):
    """A prefill continuation is excluded by `is_prefilling`, not by its token
    count, so tails wider than one token are covered by the same rule."""
    state = _mamba_hybrid_state(num_speculative_tokens=3)
    input_batch = _input_batch([2, 3], [0, 0], [True, True])

    metadata = _prepare_attn_metadata(state, monkeypatch, input_batch)

    assert metadata.num_decode_draft_tokens_cpu.tolist() == [-1, -1]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("num_sampled", "expected_value"), [(0, 1), (3, 3)])
def test_postprocess_state_scalar_with_int32_mapping(
    num_sampled: int, expected_value: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full(
        (4,), 9, dtype=torch.int32, device="cuda"
    )
    state._align_mode = False
    state.recoverssm = None
    state._mamba_ctx = None
    idx_mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled)

    expected = torch.tensor(
        [expected_value, 9, expected_value, 9], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(state.num_accepted_tokens_gpu, expected)


def test_recoverssm_commits_accepted_window_after_v2_sampling() -> None:
    state = RecoverSSMState()
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = None
    num_sampled = torch.tensor([3, 1], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    attn_groups = _recoverssm_attn_groups("layer")

    state.record_step({"layer": metadata}, attn_groups, for_capture=False)
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )

    metadata.commit_recoverssm_state.assert_called_once_with(num_sampled)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(
    ("computed", "sampled", "expected_column"),
    [(4, 3, 0), (4, 4, 0), (4, 5, 1), (8, 8, 1), (8, 9, 2)],
)
def test_recoverssm_align_tracks_mixed_batch_state_and_neutralizes_copy_bias(
    computed: int, sampled: int, expected_column: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state._mamba_ctx = None
    state._mamba_state_idx_gpu = torch.full((5,), -1, dtype=torch.int32, device="cuda")
    state.recoverssm = RecoverSSMState()
    state.num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device="cuda"
    )
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = RecoverSSMPostprocessMetadata(
        num_spec_decodes=1,
        request_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
        num_computed_tokens=torch.tensor(
            [6, computed], dtype=torch.int32, device="cuda"
        ),
        block_size=8,
        block_table=torch.zeros((2, 4), dtype=torch.int32, device="cuda"),
    )
    num_sampled = torch.tensor([2, sampled], dtype=torch.int32, device="cuda")
    idx_mapping = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    attn_groups = _recoverssm_attn_groups("layer")

    state.recoverssm.record_step({"layer": metadata}, attn_groups, for_capture=False)

    state.postprocess_state(idx_mapping, num_sampled)

    expected_state_indices = [-1, expected_column, -1, -1, -1]
    assert state._mamba_state_idx_gpu.tolist() == expected_state_indices
    expected_accepted = [9, 1, 9, 2, 9]
    assert state.num_accepted_tokens_gpu.tolist() == expected_accepted
