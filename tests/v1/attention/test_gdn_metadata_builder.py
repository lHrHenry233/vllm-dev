# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDNAttentionMetadataBuilder.build() — specifically the
reclassification of non-spec decodes as prefills when spec decodes exist.
Covers the fix for https://github.com/vllm-project/vllm/issues/34845.
"""

from dataclasses import dataclass, fields

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
    plan_flashinfer_checkpoints,
)
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


@dataclass
class GDNBuildTestCase:
    """Specification for a GDN metadata builder classification test."""

    seq_lens: list[int]
    query_lens: list[int]
    num_decode_draft_tokens: list[int] | None  # None = no spec config
    num_speculative_tokens: int
    expected_num_decodes: int
    expected_num_prefills: int
    expected_num_prefill_tokens: int
    expected_num_spec_decodes: int


GDN_BUILD_TEST_CASES = {
    # The original #34845 crash: non-spec query_len=1 + spec decode
    "mixed_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[65, 20],
        query_lens=[1, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
    # All requests are spec decodes — no reclassification needed
    "pure_spec_decode": GDNBuildTestCase(
        seq_lens=[50, 30],
        query_lens=[3, 3],
        num_decode_draft_tokens=[2, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=2,
    ),
    # Padded (CUDA graph) sequences trail the spec decodes
    "pure_spec_decode_with_padding": GDNBuildTestCase(
        seq_lens=[50, 30, 16],
        query_lens=[3, 3, 0],
        num_decode_draft_tokens=[2, 2, -1],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=2,
    ),
    # No speculative config at all — standard decode path
    "pure_regular_decode": GDNBuildTestCase(
        seq_lens=[40, 30, 20],
        query_lens=[1, 1, 1],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=0,
    ),
    # No speculative config, decode alongside prefill
    "regular_decode_with_prefill": GDNBuildTestCase(
        seq_lens=[40, 100],
        query_lens=[1, 50],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=1,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=0,
    ),
    # Multi-token prefill alongside spec decode — no decode to reclassify
    "spec_decode_with_real_prefill": GDNBuildTestCase(
        seq_lens=[100, 20],
        query_lens=[50, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # All three types in one batch — decode gets reclassified
    "prefill_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[100, 65, 20],
        query_lens=[50, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=2,
        expected_num_prefill_tokens=51,
        expected_num_spec_decodes=1,
    ),
    # Multiple non-spec query_len=1 requests all reclassified
    "multiple_decodes_reclassified": GDNBuildTestCase(
        seq_lens=[40, 50, 60, 20],
        query_lens=[1, 1, 1, 3],
        num_decode_draft_tokens=[-1, -1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=3,
        expected_num_prefill_tokens=3,
        expected_num_spec_decodes=1,
    ),
    # Zero-length padded sequence excluded from counts
    "zero_length_padding_with_spec": GDNBuildTestCase(
        seq_lens=[16, 65, 20],
        query_lens=[0, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
}


def _create_gdn_builder(
    num_speculative_tokens: int = 0,
    full_cuda_graph: bool = False,
    *,
    mamba_cache_mode: str = "none",
    kv_cache_spec: MambaSpec | None = None,
    device: torch.device = DEVICE,
    model_name: str = "Qwen/Qwen3.5-0.8B",
) -> GDNAttentionMetadataBuilder:
    """Create a GDNAttentionMetadataBuilder with minimal config."""
    vllm_config = create_vllm_config(
        model_name=model_name,
        block_size=BLOCK_SIZE,
    )
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    vllm_config.compilation_config.cudagraph_mode = (
        CUDAGraphMode.FULL_AND_PIECEWISE if full_cuda_graph else CUDAGraphMode.NONE
    )
    if num_speculative_tokens > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    mamba_spec = kv_cache_spec or MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=device,
    )


@pytest.fixture
def gdn_model_name(tmp_path):
    from transformers import Qwen3Config

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
    return str(tmp_path)


def _build(
    builder: GDNAttentionMetadataBuilder,
    batch_spec: BatchSpec,
    num_decode_draft_tokens: list[int] | None = None,
    block_table: torch.Tensor | None = None,
) -> GDNAttentionMetadata:
    """Build GDN attention metadata, optionally with spec-decode kwargs."""
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    if block_table is not None:
        common = common.replace(block_table_tensor=block_table)
    kwargs: dict = {}
    if num_decode_draft_tokens is not None:
        kwargs["num_decode_draft_tokens_cpu"] = torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        )
        kwargs["num_accepted_tokens"] = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=DEVICE
        )
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_gdn_build_classification(test_case: GDNBuildTestCase):
    """Test that GDN metadata builder classifies requests correctly."""
    builder = _create_gdn_builder(test_case.num_speculative_tokens)
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    meta = _build(builder, batch, test_case.num_decode_draft_tokens)

    assert meta.num_decodes == test_case.expected_num_decodes
    assert meta.num_prefills == test_case.expected_num_prefills
    assert meta.num_prefill_tokens == test_case.expected_num_prefill_tokens
    assert meta.num_spec_decodes == test_case.expected_num_spec_decodes
    if meta.spec_state_indices_tensor is not None:
        assert len(meta.spec_state_indices_tensor) == meta.num_spec_decodes


@pytest.mark.parametrize("mamba_cache_mode", ["none", "align"])
@pytest.mark.parametrize("full_cuda_graph", [False, True])
@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_update_block_table_matches_build(
    test_case: GDNBuildTestCase, full_cuda_graph: bool, mamba_cache_mode: str
):
    """update_block_table() on another group's metadata matches build()."""
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    src, dst, ref = (
        _create_gdn_builder(
            test_case.num_speculative_tokens,
            full_cuda_graph,
            mamba_cache_mode=mamba_cache_mode,
        )
        for _ in range(3)
    )
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    if mamba_cache_mode == "align":
        # MRV2 precomputes these for every Mamba group each step.
        dst.mamba_aligned_state_indices = ref.mamba_aligned_state_indices = (
            mamba_get_block_table_tensor(
                common.block_table_tensor, common.seq_lens, ref.kv_cache_spec, "align"
            )
        )
    draft_tokens = test_case.num_decode_draft_tokens
    expected = _build(ref, batch, draft_tokens, common.block_table_tensor)
    source = _build(src, batch, draft_tokens)
    state_fields = (
        "spec_state_indices_tensor",
        "non_spec_state_indices_tensor",
        "prefill_state_indices",
    )
    source_indices = [getattr(source, f) for f in state_fields]
    source_indices = [t if t is None else t.clone() for t in source_indices]
    meta = dst.update_block_table(
        source, common.block_table_tensor, common.slot_mapping
    )

    group_fields = {*state_fields, "checkpoint"}
    for metadata_field in fields(meta):
        if metadata_field.name not in group_fields:
            assert getattr(meta, metadata_field.name) is getattr(
                source, metadata_field.name
            )

    for field, source_index in zip(state_fields, source_indices):
        actual = getattr(meta, field)
        torch.testing.assert_close(actual, getattr(expected, field))
        # The source group's indices are untouched.
        torch.testing.assert_close(getattr(source, field), source_index)
        # FULL graph state indices land in this group's own buffers.
        if full_cuda_graph and meta.num_prefills == 0 and actual is not None:
            assert actual.data_ptr() == getattr(dst, field).data_ptr()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("state_slots", [1, 2])
def test_aligned_full_decode_reuses_live_group_indices(
    state_slots, monkeypatch, gdn_model_name
):
    """MRV2's persistent indices remain live across padded decode batches."""
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    device = torch.device("cuda")
    spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
        num_speculative_blocks=state_slots - 1,
    )
    src, dst, ref = (
        _create_gdn_builder(
            full_cuda_graph=True,
            mamba_cache_mode="align",
            kv_cache_spec=spec,
            device=device,
            model_name=gdn_model_name,
        )
        for _ in range(3)
    )
    indices = torch.empty((2, 4, state_slots), dtype=torch.int32, device=device)
    for group, builder in enumerate((src, dst)):
        assert builder.supports_update_block_table
        builder.mamba_aligned_state_indices = indices[group]

    pointers = None
    for step, num_decodes in enumerate((2, 1, 3)):
        indices.copy_(
            torch.arange(indices.numel(), device=device).view_as(indices)
            + 10
            + 100 * step
        )
        # BlockTables clears padded rows before the aligned-index kernel.
        indices[:, num_decodes:].fill_(NULL_BLOCK_ID)
        ref.mamba_aligned_state_indices = indices[1].clone()
        batch = BatchSpec(
            seq_lens=[40] * num_decodes + [0] * (4 - num_decodes),
            query_lens=[1] * num_decodes + [0] * (4 - num_decodes),
        )
        common = create_common_attn_metadata(batch, BLOCK_SIZE, device)
        source = src.build(0, common)
        actual = dst.update_block_table(
            source, common.block_table_tensor, common.slot_mapping
        )
        expected = ref.build(0, common)
        for group, metadata in enumerate((source, actual)):
            value = metadata.non_spec_state_indices_tensor
            assert value is not None
            expected_buffer = (
                indices[group]
                if state_slots == 1
                else (src, dst)[group].non_spec_state_indices_tensor
            )
            assert value.data_ptr() == expected_buffer.data_ptr()
            assert value[num_decodes:].tolist() == [NULL_BLOCK_ID] * (4 - num_decodes)
        torch.testing.assert_close(
            actual.non_spec_state_indices_tensor, expected.non_spec_state_indices_tensor
        )
        source_indices = source.non_spec_state_indices_tensor
        actual_indices = actual.non_spec_state_indices_tensor
        assert source_indices is not None
        assert actual_indices is not None
        current = (
            source_indices.data_ptr(),
            actual_indices.data_ptr(),
        )
        assert pointers is None or current == pointers
        pointers = current
        assert source_indices[0] != actual_indices[0]


@pytest.mark.parametrize("full_cuda_graph", [False, True])
@pytest.mark.parametrize("state_slots", [1, 2])
@pytest.mark.parametrize("use_spec_decode", [False, True])
def test_aligned_update_reuses_live_non_spec_view(
    full_cuda_graph, state_slots, use_spec_decode, gdn_model_name
):
    """Public aligned inputs keep their view while group-local IDs change."""
    spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
        num_speculative_blocks=state_slots - 1,
    )
    source_builder, builder = (
        _create_gdn_builder(
            int(use_spec_decode),
            full_cuda_graph,
            mamba_cache_mode="align",
            kv_cache_spec=spec,
            model_name=gdn_model_name,
        )
        for _ in range(2)
    )
    common = create_common_attn_metadata(
        BatchSpec([40, 40], [1, 1]), BLOCK_SIZE, DEVICE
    )
    source_builder.mamba_aligned_state_indices = torch.tensor(
        [[91] * state_slots, [92] * state_slots], dtype=torch.int32
    )
    source = source_builder.build(0, common)
    source_indices = source.non_spec_state_indices_tensor
    assert source_indices is not None
    original_source = source_indices.clone()
    backing = torch.tensor([[11] * state_slots, [21] * state_slots], dtype=torch.int32)
    builder.mamba_aligned_state_indices = backing
    first = builder.update_block_table(
        source, common.block_table_tensor, common.slot_mapping
    )
    backing.copy_(torch.tensor([[31] * state_slots, [41] * state_slots]))
    second = builder.update_block_table(
        source, common.block_table_tensor, common.slot_mapping
    )
    first_indices = first.non_spec_state_indices_tensor
    second_indices = second.non_spec_state_indices_tensor
    assert first_indices is not None
    assert second_indices is not None
    if not full_cuda_graph:
        assert second_indices is first_indices
    else:
        assert (
            second_indices.data_ptr()
            == builder.non_spec_state_indices_tensor.data_ptr()
        )
    assert second_indices.tolist() == [31, 41]
    assert second is not first and second is not source
    torch.testing.assert_close(source.non_spec_state_indices_tensor, original_source)

    short_common = create_common_attn_metadata(BatchSpec([40], [1]), BLOCK_SIZE, DEVICE)
    source_builder.mamba_aligned_state_indices = torch.tensor(
        [[93] * state_slots], dtype=torch.int32
    )
    short = source_builder.build(0, short_common)
    builder.mamba_aligned_state_indices = torch.tensor(
        [[51] * state_slots], dtype=torch.int32
    )
    third = builder.update_block_table(
        short, short_common.block_table_tensor, short_common.slot_mapping
    )
    third_indices = third.non_spec_state_indices_tensor
    first_indices = first.non_spec_state_indices_tensor
    assert third_indices is not None
    assert first_indices is not None
    assert third_indices.tolist() == [51]
    assert third_indices is not first_indices
    if not full_cuda_graph:
        assert first_indices.tolist() == [31, 41]


@pytest.mark.parametrize("cache_mode", ["none", "align"])
def test_aligned_update_preserves_mixed_spec_indices(cache_mode, gdn_model_name):
    """Real mixed metadata preserves masks and independent state gathers."""
    builder = _create_gdn_builder(
        num_speculative_tokens=1,
        full_cuda_graph=True,
        mamba_cache_mode=cache_mode,
        model_name=gdn_model_name,
    )
    table = torch.tensor([[11, 12], [21, 22]], dtype=torch.int32)
    common = create_common_attn_metadata(
        BatchSpec([20, 1], [2, 1]), BLOCK_SIZE, DEVICE
    ).replace(block_table_tensor=table, is_prefilling=torch.tensor([False, True]))
    builder.mamba_aligned_state_indices = table
    source = builder.build(
        0,
        common,
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([1, -1], dtype=torch.int32),
    )
    assert source.num_prefills == source.num_spec_decodes == 1
    first = builder.update_block_table(source, table, common.slot_mapping)
    table.copy_(torch.tensor([[31, 32], [41, 42]], dtype=torch.int32))
    second = builder.update_block_table(source, table, common.slot_mapping)
    second_spec_indices = second.spec_state_indices_tensor
    second_non_spec_indices = second.non_spec_state_indices_tensor
    second_prefill_indices = second.prefill_state_indices
    first_spec_indices = first.spec_state_indices_tensor
    assert second_spec_indices is not None
    assert second_non_spec_indices is not None
    assert second_prefill_indices is not None
    assert first_spec_indices is not None
    assert second_spec_indices.tolist() == [[31, 32]]
    assert second_non_spec_indices.tolist() == [41]
    assert second_prefill_indices.tolist() == [41]
    assert first_spec_indices.tolist() == [[11, 12]]
    assert second.spec_sequence_masks_cpu is source.spec_sequence_masks_cpu


def test_has_initial_state_after_reclassification():
    """After reclassification, num_prefills > 0 so the prefill kernel path
    should compute has_initial_state. For the reclassified request with
    context_lens > 0, the corresponding entry must be True."""
    builder = _create_gdn_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[65, 20], query_lens=[1, 3])
    meta = _build(builder, batch, num_decode_draft_tokens=[-1, 2])

    assert meta.num_prefills > 0, "reclassification should produce prefills"
    assert meta.has_initial_state is not None
    # req0 has context_lens = 65 - 1 = 64 > 0, so has_initial_state[0] = True
    assert meta.has_initial_state[0].item() is True


def test_full_cudagraph_spec_metadata_uses_request_count():
    """FULL cudagraph token padding must not pad request-indexed metadata."""
    num_speculative_tokens = 3
    builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        full_cuda_graph=True,
    )
    batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    meta = _build(builder, batch, num_decode_draft_tokens=[3, 3])

    assert meta.num_spec_decodes == batch.batch_size
    assert meta.num_spec_decode_tokens == batch.compute_num_tokens()
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.shape == (
        batch.batch_size,
        num_speculative_tokens + 1,
    )
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.shape == (batch.batch_size,)
    assert meta.spec_query_start_loc is not None
    assert meta.spec_query_start_loc.shape == (batch.batch_size + 1,)
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.shape == (batch.batch_size,)


def _build_non_spec(
    batch: BatchSpec,
    is_prefilling: list[bool] | None,
    full_cuda_graph: bool = False,
):
    common_attn_metadata = create_common_attn_metadata(
        batch, BLOCK_SIZE, DEVICE
    ).replace(
        is_prefilling=None
        if is_prefilling is None
        else torch.tensor(is_prefilling, dtype=torch.bool)
    )
    builder = _create_gdn_builder(full_cuda_graph=full_cuda_graph)
    return builder, common_attn_metadata, builder.build(0, common_attn_metadata)


@pytest.mark.parametrize(
    ("seq_len", "query_len", "is_prefilling", "num_prefills"),
    [
        pytest.param(1, 1, True, 1, id="first-chunk"),
        pytest.param(65, 1, True, 0, id="resumed-chunk"),
        pytest.param(0, 0, True, 0, id="padding"),
        pytest.param(1, 1, None, 0, id="missing-prefill-flag"),
    ],
)
def test_one_token_chunk_classification(
    seq_len: int,
    query_len: int,
    is_prefilling: bool | None,
    num_prefills: int,
):
    """Only a real first chunk with a prefill flag needs state initialization."""
    _, _, meta = _build_non_spec(
        BatchSpec(seq_lens=[100, seq_len], query_lens=[1, query_len]),
        is_prefilling=None if is_prefilling is None else [False, is_prefilling],
    )

    assert meta.num_prefills == num_prefills
    assert meta.num_decodes == 2 - num_prefills
    assert meta.num_prefill_tokens == num_prefills
    assert meta.num_decode_tokens == 1 + query_len - num_prefills
    if num_prefills:
        assert meta.has_initial_state is not None
        assert meta.has_initial_state.tolist() == [True, False]
    else:
        assert meta.has_initial_state is None


def test_one_token_first_chunk_excludes_padding():
    """Neither padding requests nor padding tokens count as prefill work."""
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[100, 1, 0, 0], query_lens=[1, 1, 0, 0]),
        BLOCK_SIZE,
        DEVICE,
    ).replace(
        is_prefilling=torch.tensor([False, True, False, False], dtype=torch.bool),
        num_actual_tokens=4,
    )
    meta = _create_gdn_builder().build(0, common)

    assert meta.num_decodes == 1
    assert meta.num_prefills == 1
    assert meta.num_decode_tokens == 1
    assert meta.num_prefill_tokens == 1


def test_cudagraph_capture_batch_stays_decode_only():
    """Capture rows have no history, but must still select decode kernels."""
    batch = BatchSpec(seq_lens=[1] * 4, query_lens=[1] * 4)
    builder, common_attn_metadata, _ = _build_non_spec(
        batch, [False] * 4, full_cuda_graph=True
    )
    meta = builder.build_for_cudagraph_capture(common_attn_metadata)

    assert meta.num_prefills == 0
    assert meta.num_decodes == 4
    assert meta.has_initial_state is None
    staged = meta.non_spec_state_indices_tensor
    assert staged is not None
    assert staged.data_ptr() == builder.non_spec_state_indices_tensor.data_ptr()
    torch.testing.assert_close(staged, common_attn_metadata.block_table_tensor[:, 0])


def _create_checkpoint_builder_and_batch(
    num_spec: int, prefix_match_unit: int | None = 16
) -> tuple[GDNAttentionMetadataBuilder, CommonAttentionMetadata, dict]:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
    )
    if num_spec > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_spec,
        )
    vllm_config.cache_config.mamba_cache_mode = "align"
    vllm_config.cache_config.prefix_match_unit = prefix_match_unit
    mamba_spec = MambaSpec(
        block_size=64,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
        num_speculative_blocks=num_spec,
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=16,
    )
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )
    vllm_config.cache_config.hash_block_size = prefix_match_unit or 64
    vllm_config.cache_config.cache_hit_alignment_tokens = prefix_match_unit or 64
    batch = BatchSpec(
        seq_lens=[65, 100, 100], query_lens=[3 if num_spec else 1, 100, 99]
    )
    common = create_common_attn_metadata(batch, 64, DEVICE, arange_block_indices=True)
    table = common.block_table_tensor + 1
    if num_spec:
        table = torch.cat([table, torch.zeros(3, num_spec, dtype=table.dtype)], dim=1)
    common = common.replace(block_table_tensor=table)
    kwargs = {}
    if num_spec:
        kwargs = dict(
            num_decode_draft_tokens_cpu=torch.tensor([2, -1, -1], dtype=torch.int32),
            num_accepted_tokens=torch.ones(3, dtype=torch.int32),
        )
    return builder, common, kwargs


@pytest.mark.parametrize("num_spec", [0, 3])
@pytest.mark.parametrize(
    "prefix_match_unit, expected_offset", [(None, 64), (16, 96), (8, 96)]
)
def test_checkpoint_metadata_preserves_non_spec_order(
    num_spec, prefix_match_unit, expected_offset
):
    """Only eligible non-spec rows get a checkpoint in their reserved page."""
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    builder, common, kwargs = _create_checkpoint_builder_and_batch(
        num_spec, prefix_match_unit
    )
    actual = builder.build(0, common, **kwargs)
    assert actual.checkpoint is not None
    offsets = [expected_offset, 0] if num_spec else [0, expected_offset, 0]
    slots = [common.block_table_tensor[1, 0].item(), NULL_BLOCK_ID]
    if not num_spec:
        slots.insert(0, NULL_BLOCK_ID)
    torch.testing.assert_close(
        actual.checkpoint.checkpoint_offsets, torch.tensor(offsets, dtype=torch.int32)
    )
    torch.testing.assert_close(
        actual.checkpoint.state_indices, torch.tensor(slots, dtype=torch.int32)
    )


@pytest.mark.parametrize("num_spec", [0, 3])
def test_update_block_table_regathers_checkpoint(num_spec):
    """Checkpoint pages come from the target group's block table, not the
    source group's (otherwise the target group's checkpoint page is never
    written while the prefix cache treats it as valid)."""
    src, common, kwargs = _create_checkpoint_builder_and_batch(num_spec)
    dst, _, _ = _create_checkpoint_builder_and_batch(num_spec)
    ref, _, _ = _create_checkpoint_builder_and_batch(num_spec)
    other_table = common.block_table_tensor + 100
    other = common.replace(block_table_tensor=other_table)
    dst.mamba_aligned_state_indices = mamba_get_block_table_tensor(
        other_table, other.seq_lens, dst.kv_cache_spec, "align"
    )

    source = src.build(0, common, **kwargs)
    expected = ref.build(0, other, **kwargs)
    actual = dst.update_block_table(source, other_table, other.slot_mapping)

    assert source.checkpoint is not None and expected.checkpoint is not None
    assert actual.checkpoint is not None
    torch.testing.assert_close(
        actual.checkpoint.state_indices, expected.checkpoint.state_indices
    )
    torch.testing.assert_close(
        actual.checkpoint.checkpoint_offsets, expected.checkpoint.checkpoint_offsets
    )
    assert not torch.equal(
        actual.checkpoint.state_indices, source.checkpoint.state_indices
    )


def test_plan_flashinfer_checkpoints_selects_the_state_at_each_offset():
    """Each checkpoint row maps to the state FlashInfer emits at its offset.

    The leading decode row is not in the prefill kernel and never checkpoints;
    the 50-token prefill is shorter than the stride and emits no states.
    """
    every_n_tokens, cu_starts, rows = plan_flashinfer_checkpoints(
        offsets=[0, 128, 0, 192], query_lens=[200, 50, 300]
    )

    assert every_n_tokens == 64
    assert cu_starts == [0, 3, 3, 7]
    # Row 1: state after 128 tokens of the first prefill. Row 3: after 192
    # tokens of the third prefill, whose states start at index 3.
    assert rows == [0, 1, 0, 5]
