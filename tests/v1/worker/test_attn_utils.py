# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Padded-page handling in create_kv_cache_views.

Guards that a page_size_padded spec strides the block dimension by the padded page
while keeping per-block content compact, so padding bytes at the end of each page are
never addressed by the logical view.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import vllm.v1.hisparse.binding as attn_utils_module
import vllm.v1.hisparse.runtime as hisparse_runtime_module
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    dense_kv_cache_views,
)
from vllm.config import KVTransferConfig
from vllm.config.attention import HiSparseConfig
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend, AttentionCGSupport
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.hisparse.binding import allocate_hisparse_kv_caches
from vllm.v1.hisparse.runtime import HiSparseCacheHandle, HiSparseRuntime
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SparseCacheRole,
    SparseFullAttentionSpec,
    UniformTypeKVCacheSpecs,
    compute_layout_strides,
)
from vllm.v1.worker.gpu import attn_utils
from vllm.v1.worker.gpu.attn_utils import (
    FastPrefillHelper,
    get_attn_cg_support,
    get_query_lens_mismatch_unsupported_backend,
)
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.utils import (
    AttentionGroup,
    allocate_kv_cache,
    copy_kv_cache_blocks_inplace,
    customize_attention_spec,
    map_kv_caches_to_kernel_blocks,
)


@pytest.mark.parametrize("num_groups", [0, 2])
@pytest.mark.parametrize("num_tokens", [0, 4])
def test_slot_mapping_rows_preserve_layer_aliases(num_groups, num_tokens):
    """Tuple and Tensor inputs expose the same live rows, including empty views."""
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32
    )
    config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer.{i}", f"shared.{i}"], spec)
            for i in range(num_groups)
        ],
    )
    slots = torch.arange(num_groups * 8, dtype=torch.int64).view(num_groups, 8)
    view = slots[:, :num_tokens]
    rows = view.unbind(0)
    actual = attn_utils.build_slot_mappings_by_layer(rows, config)
    reference = attn_utils.build_slot_mappings_by_layer(view, config)
    slots.add_(17)
    assert actual.keys() == reference.keys()
    for i, row in enumerate(rows):
        assert actual[f"layer.{i}"] is row
        assert actual[f"shared.{i}"] is row
        torch.testing.assert_close(actual[f"layer.{i}"], reference[f"layer.{i}"])


@pytest.mark.parametrize("sparse_full_attention", [False, True])
@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize(
    "scenario", ["intersection", "supported", "multiple", "no-factor", "disabled"]
)
def test_get_kv_cache_spec_resolves_hisparse_block_size(
    sparse_full_attention, draft, scenario
):
    """Real backend geometry preserves compression, draft tags and ownership."""
    from tests.v1.core.test_kv_cache_planning import _hisparse_planning_config
    from vllm.models.qwen4_exp.common.qsa_cache import QSAStateBackend

    # Follow model registration order before loading its QSA submodule.
    from vllm.models.qwen4_exp.nvidia import model  # noqa: F401
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionBackend
    from vllm.v1.attention.backends.mla.flashattn_mla import FlashAttnMLABackend
    from vllm.v1.attention.backends.mla.flashmla import FlashMLABackend
    from vllm.v1.worker.utils import select_common_block_size

    enabled = scenario != "disabled"
    block_size = 256 if scenario in ("intersection", "disabled") else 64
    if scenario == "no-factor":
        block_size = 8 if sparse_full_attention else 32
    main_backend = (
        Qwen4ExpQSAFlashAttentionBackend if sparse_full_attention else FlashMLABackend
    )
    if scenario == "multiple" and not sparse_full_attention:
        main_backend = FlashAttnMLABackend
    indexer_backend = QSAStateBackend if sparse_full_attention else FlashAttnMLABackend
    main = (
        SparseFullAttentionSpec(
            block_size=block_size,
            num_kv_heads=2,
            total_num_kv_heads=4,
            head_size=32,
            dtype=torch.bfloat16,
            top_k=35,
            is_draft=draft,
        )
        if sparse_full_attention
        else MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            is_draft=draft,
        )
    )
    ratio = 4 if sparse_full_attention else 1
    indexer = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=ratio,
        cache_role=SparseCacheRole.INDEXER,
        is_draft=draft,
    )
    specs = {"main": main, "indexer": indexer}
    if scenario == "no-factor":
        with pytest.raises(ValueError, match="block size"):
            select_common_block_size(
                block_size, [main_backend, indexer_backend], list(specs.values())
            )
        assert main.block_size == indexer.block_size == block_size
        return
    resolved_size = (
        select_common_block_size(
            block_size, [main_backend, indexer_backend], list(specs.values())
        )
        if enabled
        else block_size
    )
    expected = (
        64 if scenario == "intersection" and not sparse_full_attention else block_size
    )
    assert resolved_size == expected
    resolved = {
        name: spec.copy_with_new_block_size(resolved_size)
        for name, spec in specs.items()
    }
    assert all(spec.block_size == expected for spec in resolved.values())
    assert resolved["indexer"].tokens_per_state == ratio
    assert resolved["indexer"].num_states == expected // ratio
    assert resolved["main"].is_draft == resolved["indexer"].is_draft == draft
    assert all(spec.block_size == block_size for spec in specs.values())
    if sparse_full_attention:
        main_spec = resolved["main"]
        assert isinstance(main_spec, SparseFullAttentionSpec)
        assert main_spec.top_k == 35
        assert main_spec.total_num_kv_heads == 4
    config = _hisparse_planning_config()
    config.model_config.hf_text_config.update({"index_topk": 128})
    config.attention_config.hisparse_config = HiSparseConfig() if enabled else None
    groups = attn_utils_module.get_hisparse_kv_cache_groups(config, resolved)
    if enabled:
        assert groups is not None
        names = [name for group in groups for name in group.layer_names]
        assert len(names) == len(set(names)) == len(specs) + 2
        assert set(names) == {
            "main",
            "indexer",
            "main.hisparse_resident",
            "main.hisparse_hot",
        }
        auxiliary = {
            name: group.kv_cache_spec
            for group in groups
            for name in group.layer_names
            if name.endswith((".hisparse_resident", ".hisparse_hot"))
        }
        assert (
            auxiliary["main.hisparse_resident"].page_size_bytes
            == resolved["main"].page_size_bytes
        )
        assert (
            auxiliary["main.hisparse_hot"].page_size_bytes
            == resolved["main"].page_size_bytes
        )
        hot_spec = auxiliary["main.hisparse_hot"]
        assert isinstance(hot_spec, HiSparseHotSpec)
        assert hot_spec.blocks_per_request > 0
    else:
        assert groups is None
        assert resolved == specs


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(90),
    reason="real FlashMLA sparse layer requires CUDA capability >= 9.0",
)
@pytest.mark.parametrize("enabled", [False, True])
def test_get_kv_cache_spec_real_mla_caller_preserves_dense_customization(
    tmp_path, dist_init, workspace_init, enabled
):
    """Discover real owners, resolve sparse blocks and retain dense/off specs."""
    from dataclasses import replace

    from transformers import DeepseekV3Config

    from tests.v1.attention.utils import create_vllm_config
    from vllm.config import AttentionConfig, set_current_vllm_config
    from vllm.model_executor.layers.attention import Attention
    from vllm.models.deepseek_v32.attention import DeepseekV32Attention
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend

    hf = DeepseekV3Config(
        architectures=["DeepseekV32ForCausalLM"],
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=512,
        q_lora_rank=64,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        index_head_dim=128,
        index_n_heads=4,
        index_topk=4,
        max_position_embeddings=1024,
        vocab_size=128,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    hf.save_pretrained(tmp_path)
    config = create_vllm_config(
        model_name=str(tmp_path),
        dtype="bfloat16",
        block_size=256,
        max_model_len=512,
        max_num_batched_tokens=16,
        max_num_seqs=2,
    )
    config.attention_config = AttentionConfig(
        hisparse_config=HiSparseConfig() if enabled else None,
        sparse_mla_prefill_workspace_factor=2,
    )
    config.kv_transfer_config = (
        KVTransferConfig(
            kv_connector="HiSparseConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"host_pool_gib": 1 / 1024},
        )
        if enabled
        else None
    )
    config.cache_config.kv_cache_layout = "BLNHC"
    config.scheduler_config.async_scheduling = False
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        logical_topk = torch.empty((16, hf.index_topk), dtype=torch.int32)
        sparse = DeepseekV32Attention(
            config,
            hf,
            prefix="model.layers.0.self_attn",
            topk_indices_buffer=logical_topk,
            attn_backend=FlashMLASparseBackend,
        )
        dense = Attention(
            4,
            128,
            128**-0.5,
            num_kv_heads=4,
            cache_config=config.cache_config,
            prefix="model.layers.1.self_attn",
            attn_backend=FlashAttentionBackend,
        )
    assert sparse.indexer is not None
    indexer = sparse.indexer.k_cache
    owners = {
        sparse.layer_name: sparse,
        indexer.prefix: indexer,
        "model.layers.1.self_attn": dense,
    }
    assert all(
        config.compilation_config.static_forward_context[name] is owner
        for name, owner in owners.items()
    )
    original: dict[str, AttentionSpec] = {}
    for name, owner in owners.items():
        owner_spec = owner.get_kv_cache_spec(config)
        assert isinstance(owner_spec, AttentionSpec)
        original[name] = owner_spec
    assert all(
        spec.block_size == 256 and not spec.is_draft for spec in original.values()
    )
    # The runner marks real draft owners through this public layer flag.
    sparse.is_draft_layer = indexer.is_draft_layer = True
    resolved = attn_utils.get_kv_cache_spec(config)
    expected_block = 64 if enabled else 256
    for name in (sparse.layer_name, indexer.prefix):
        assert resolved[name].block_size == expected_block
        assert resolved[name].is_draft
    assert resolved[indexer.prefix].tokens_per_state == 1
    assert resolved[indexer.prefix].num_states == expected_block
    dense_name = "model.layers.1.self_attn"
    assert not resolved[dense_name].is_draft
    assert resolved[dense_name] == customize_attention_spec(
        dense.get_attn_backend(), original[dense_name]
    )
    assert config.cache_config.block_size == 256
    assert all(
        spec.block_size == 256 and not spec.is_draft for spec in original.values()
    )
    if enabled:
        resident_name = f"{sparse.layer_name}.hisparse_resident"
        hot_name = f"{sparse.layer_name}.hisparse_hot"
        assert set(resolved) == set(owners) | {resident_name, hot_name}
        resident, hot = resolved[resident_name], resolved[hot_name]
        assert isinstance(resident, HiSparseResidentSpec)
        assert isinstance(hot, HiSparseHotSpec)
        assert resident.block_size == hot.block_size == 64
        assert (
            resident.page_size_bytes
            == hot.page_size_bytes
            == resolved[sparse.layer_name].page_size_bytes
        )
    else:
        assert resolved == {
            name: replace(
                customize_attention_spec(owner.get_attn_backend(), original[name]),
                is_draft=owner.is_draft_layer,
            )
            for name, owner in owners.items()
        }


def test_get_kv_cache_spec_tags_draft_layers(monkeypatch):
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
    )
    backend = SimpleNamespace(
        customize_spec=AttentionBackend.customize_spec,
        get_supported_kernel_block_sizes=lambda _=None: [16],
    )
    layers = {
        name: SimpleNamespace(
            get_kv_cache_spec=lambda _: spec,
            get_attn_backend=lambda: backend,
            is_draft_layer=name == "draft",
        )
        for name in ("target", "draft")
    }
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_: layers)
    config = SimpleNamespace(attention_config=SimpleNamespace(hisparse_config=None))

    specs = attn_utils.get_kv_cache_spec(config)
    assert specs["draft"].is_draft and not specs["target"].is_draft


class _FakeMetadataBuilder:
    def __init__(self, support: AttentionCGSupport, varlen_bound: int | None = None):
        self.support = support
        self.varlen_bound = varlen_bound

    def get_cudagraph_support(self, *_args):
        return self.support

    def get_varlen_cudagraph_max_query_len(self, *_args):
        return self.varlen_bound


class _TargetBackend:
    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return True


class _DraftBackend:
    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False


def test_attention_checks_preserve_global_and_target_scoped_support():
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    target_group = AttentionGroup(
        _TargetBackend,
        ["target"],
        spec,
        0,
    )
    target_group.metadata_builders = [
        _FakeMetadataBuilder(AttentionCGSupport.ALWAYS)  # type: ignore[list-item]
    ]
    draft_group = AttentionGroup(
        _DraftBackend,
        ["draft"],
        spec,
        0,
    )
    draft_group.metadata_builders = [
        _FakeMetadataBuilder(AttentionCGSupport.UNIFORM_BATCH)  # type: ignore[list-item]
    ]
    groups = [[target_group, draft_group]]

    # The runner-wide execution mode must still honor the drafter's limit.
    unfiltered = get_attn_cg_support(groups, None)
    assert unfiltered.min_cg_support == AttentionCGSupport.UNIFORM_BATCH
    assert unfiltered.min_cg_attn_backend == "_DraftBackend"

    # Adaptive verification validates only the target's varlen graphs.
    target_only = get_attn_cg_support(
        groups,
        None,
        checked_layer_names={"target"},
    )
    assert target_only.min_cg_support == AttentionCGSupport.ALWAYS
    assert target_only.min_cg_attn_backend is None
    assert (
        get_query_lens_mismatch_unsupported_backend(
            groups,
            checked_layer_names={"target"},
        )
        is None
    )

    # Shared target/draft groups still participate in target-scoped checks.
    draft_group.layer_names.append("target")
    target_with_shared_group = get_attn_cg_support(
        groups,
        None,
        checked_layer_names={"target"},
    )
    assert target_with_shared_group.min_cg_support == AttentionCGSupport.UNIFORM_BATCH
    assert (
        get_query_lens_mismatch_unsupported_backend(
            groups,
            checked_layer_names={"target"},
        )
        == "_DraftBackend"
    )


def test_varlen_cudagraph_unsupported_backend_checks_scoped_bounds():
    """ALWAYS passes without a bound, other builders need one at least as wide as
    the requested length, and NEVER fails whatever bound a builder reports."""
    config: Any = SimpleNamespace()
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )

    def group(
        backend: Any,
        layer_name: str,
        support: AttentionCGSupport,
        bound: int | None = None,
    ):
        builder: Any = _FakeMetadataBuilder(support, bound)
        attn_group = AttentionGroup(backend, [layer_name], spec, 0)
        attn_group.metadata_builders = [builder]
        return attn_group

    target = group(_TargetBackend, "target", AttentionCGSupport.ALWAYS)
    draft = group(_DraftBackend, "draft", AttentionCGSupport.UNIFORM_BATCH, 8)
    never = group(_DraftBackend, "never", AttentionCGSupport.NEVER, 8)

    unsupported = attn_utils.get_varlen_cudagraph_unsupported_backend
    assert unsupported([[target, draft]], config, 8) is None
    assert unsupported([[target, draft]], config, 9) == ("_DraftBackend", 8)
    assert (
        unsupported([[target, draft]], config, 9, checked_layer_names={"target"})
        is None
    )
    assert unsupported([[target, never]], config, 1) == ("_DraftBackend", None)


@pytest.mark.parametrize("num_speculative_tokens", [0, 7])
def test_flashinfer_sparse_full_graphs_exclude_prefill_keep_varlen_decode(
    num_speculative_tokens,
):
    """Prefill must fall back without disabling adaptive verification graphs."""
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseTRTLLMBackend,
    )

    backend = FlashInferMLASparseTRTLLMBackend
    config: Any = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens, parallel_drafting=False
        ),
        use_v2_model_runner=True,
    )
    spec = MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
    )
    group = AttentionGroup(backend, ["target"], spec, 0)
    group.metadata_builders = [object.__new__(backend.get_builder_cls())]
    groups = [[group]]
    support = get_attn_cg_support(groups, config)
    compilation = CompilationConfig(cudagraph_mode=CUDAGraphMode.FULL)
    mode = compilation.resolve_cudagraph_mode_and_sizes(
        support.min_cg_support,
        support.min_cg_attn_backend,
        uniform_decode_query_len=1 + num_speculative_tokens,
        use_v2_model_runner=True,
    )
    assert mode.mixed_mode() != CUDAGraphMode.FULL
    assert mode.decode_mode() == CUDAGraphMode.FULL
    assert get_query_lens_mismatch_unsupported_backend(groups) is None
    unsupported = attn_utils.get_varlen_cudagraph_unsupported_backend
    assert unsupported(groups, config, 1 + num_speculative_tokens) is None
    assert unsupported(groups, config, 2 + num_speculative_tokens) == (
        backend.__name__,
        1 + num_speculative_tokens,
    )


def test_get_kv_sharing_fast_prefill_eligible_layers(monkeypatch: pytest.MonkeyPatch):
    """Fast prefill applies to the contiguous suffix of KV-sharing layers.

    Draft-model layers register after the target model's and may share KV, so
    they must not extend (or break) the target's eligible suffix.
    """

    def check(
        layer_names: list[str],
        shared: dict[str, str],
        draft_layer_names: set[str] | None = None,
    ) -> set[str]:
        monkeypatch.setattr(
            attn_utils,
            "get_layers_from_vllm_config",
            lambda *a, **k: {name: None for name in layer_names},
        )
        monkeypatch.setattr(attn_utils, "get_shared_kv_cache_layers", lambda *a: shared)
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(kv_sharing_fast_prefill=True)
        )
        return attn_utils.get_kv_sharing_fast_prefill_eligible_layers(
            vllm_config, draft_layer_names
        )

    # No KV sharing: nothing is eligible.
    assert check(["t0", "t1"], {}) == set()

    # Trailing run of sharing layers (YOCO-style second half).
    assert check(["t0", "t1", "t2", "t3"], {"t2": "t1", "t3": "t1"}) == {"t2", "t3"}

    # A non-sharing layer after a sharing one breaks the suffix.
    assert check(["t0", "t1", "t2", "t3"], {"t1": "t0", "t3": "t0"}) == {"t3"}

    # KV-sharing draft layers at the end are collected without an exclusion...
    assert check(
        ["t0", "t1", "t2", "t3", "d0", "d1"],
        {"t2": "t1", "t3": "t1", "d0": "t1", "d1": "t1"},
    ) == {"t2", "t3", "d0", "d1"}

    # ...so the runner excludes them: skipped, not collected, and they do not
    # break the target's trailing run.
    assert check(
        ["t0", "t1", "t2", "t3", "d0", "d1"],
        {"t2": "t1", "t3": "t1", "d0": "t1", "d1": "t1"},
        draft_layer_names={"d0", "d1"},
    ) == {"t2", "t3"}

    # Feature flag off: nothing is eligible even with sharing layers.
    monkeypatch.setattr(
        attn_utils, "get_layers_from_vllm_config", lambda *a, **k: {"t0": None}
    )
    monkeypatch.setattr(
        attn_utils, "get_shared_kv_cache_layers", lambda *a: {"t0": "t0"}
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False)
    )
    assert attn_utils.get_kv_sharing_fast_prefill_eligible_layers(vllm_config) == set()


@pytest.mark.parametrize(
    "num_tokens", [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128]
)
@pytest.mark.parametrize("num_active_loras", [1, 2, 4])
def test_fast_prefill_dispatch_preserves_active_lora_count(
    num_tokens: int, num_active_loras: int
):
    """Fast-prefill padding must match the main dispatch's LoRA variant."""

    class FakeCudaGraphManager:
        device = "cpu"

        def __init__(self):
            self.dispatch_calls = []

        def dispatch(self, **kwargs):
            self.dispatch_calls.append(kwargs)
            # A captured no-LoRA graph is padded to 8 tokens, while the
            # active-LoRA path stays eager at the unpadded token count.
            if kwargs["num_active_loras"] == 0:
                num_tokens = 8
                mode = CUDAGraphMode.PIECEWISE
            else:
                num_tokens = kwargs["num_tokens"]
                mode = CUDAGraphMode.NONE
            return BatchExecutionDescriptor(
                cg_mode=mode,
                num_tokens=num_tokens,
                num_reqs=kwargs["num_reqs"],
                num_active_loras=kwargs["num_active_loras"],
                num_ubatches=1,
            )

    manager = FakeCudaGraphManager()
    helper = FastPrefillHelper(manager, max_num_tokens=max(32, num_tokens))
    metadata = helper.prepare(
        torch.arange(num_tokens, dtype=torch.int32),
        num_reqs=1,
        cu_num_logits_np=np.array([0, num_tokens], dtype=np.int32),
        has_prefill=True,
        batch_desc=BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=1,
            num_active_loras=num_active_loras,
            num_ubatches=1,
        ),
        num_active_loras=num_active_loras,
    )

    assert metadata is not None
    assert metadata.num_logits_indices == num_tokens
    assert metadata.logits_indices_padded.shape[0] == num_tokens
    assert metadata.max_logits_per_req == num_tokens
    assert manager.dispatch_calls[-1]["num_active_loras"] == num_active_loras


class _FakeSharedHostRegion:
    def __init__(self) -> None:
        self.cleanup_calls = 0
        self.base_tensor = torch.empty(1, dtype=torch.int8)

    def cleanup(self) -> None:
        self.cleanup_calls += 1


def test_profiling_cleanup_keeps_host_pool_for_real_kv_cache(monkeypatch):
    """The real KV cache must reuse the profiling host pool, not pin it again."""
    pinned = torch.empty(8, dtype=torch.int8)
    allocate_pinned = MagicMock(return_value=(pinned, pinned))
    monkeypatch.setattr(
        hisparse_runtime_module, "allocate_pinned_host_pool", allocate_pinned
    )
    monkeypatch.setattr(
        hisparse_runtime_module, "check_hisparse_host_memory", lambda size: None
    )
    vllm_config = SimpleNamespace(instance_id="profiling-reuse")
    kv_cache_config = SimpleNamespace(hisparse_shared_host_pool=False)
    pool = attn_utils_module.get_hisparse_host_pool(vllm_config, kv_cache_config)
    backing = pool.allocate(8)
    runtime = SimpleNamespace(
        _host_cache=backing, registered_host_pool=pinned, hot_backing=object()
    )
    forward_context = {
        "layer": SimpleNamespace(hisparse_cache=SimpleNamespace(runtime=runtime))
    }

    attn_utils_module.release_hisparse_profiling_cache(forward_context)

    assert not hasattr(runtime, "_host_cache")
    reused = attn_utils_module.get_hisparse_host_pool(vllm_config, kv_cache_config)
    assert reused is pool
    assert reused.allocate(8) is backing
    allocate_pinned.assert_called_once_with(8)
    hisparse_runtime_module.drop_hisparse_host_pool(vllm_config)


@pytest.mark.parametrize("failure_phase", ["allocation", "binding", "buffers"])
@pytest.mark.parametrize("shared", [False, True])
def test_init_hisparse_rolls_back_host_pool(monkeypatch, failure_phase, shared):
    """Failed initialization must release shared or private registrations."""
    region = _FakeSharedHostRegion()
    registered = torch.empty(4096, dtype=torch.int8)
    cudart = MagicMock()
    cudart.cudaHostUnregister.return_value.value = 0
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        instance_id="rollback",
    )
    pools = []

    def allocate(*args):
        pools.append(args[-1])
        if shared:
            args[-1].shared_region = region
        else:
            args[-1].registered = registered
            args[-1].backing = registered[:8]
        if failure_phase == "allocation":
            raise RuntimeError("initialization failed")
        return {}

    def bind(**kwargs):
        if failure_phase == "binding":
            raise RuntimeError("initialization failed")
        return []

    def buffers(*args, **kwargs):
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(attn_utils_module, "allocate_hisparse_kv_caches", allocate)
    monkeypatch.setattr(attn_utils_module, "bind_hisparse_kv_caches", bind)
    monkeypatch.setattr(
        attn_utils_module, "initialize_hisparse_runtime_buffers", buffers
    )
    with pytest.raises(RuntimeError, match="initialization failed"):
        attn_utils_module.init_hisparse_kv_cache(
            SimpleNamespace(),
            torch.device("cpu"),
            [],
            vllm_config,
            {},
            SimpleNamespace(),
        )
    assert region.cleanup_calls == int(shared)
    if shared:
        cudart.cudaHostUnregister.assert_not_called()
    else:
        cudart.cudaHostUnregister.assert_called_once_with(registered.data_ptr())
    assert attn_utils_module.get_hisparse_host_pool(vllm_config, None) is not pools[0]
    hisparse_runtime_module.drop_hisparse_host_pool(vllm_config)


def test_reshape_padded_kv_cache_strides_by_padded_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 256

    raw = torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    (kv_cache,) = dense_kv_cache_views(raw, spec, num_blocks, 1, KVCacheLayout.LBHNC)

    elem_size = 4  # float32
    # Content dim packs K and V: 2 * head_size.
    assert kv_cache.shape == (num_blocks, 1, 16, 2 * spec.head_size)
    assert kv_cache.dtype == spec.dtype
    assert spec.page_size_padded is not None
    assert kv_cache.stride(0) == spec.page_size_padded // elem_size
    assert kv_cache[1].storage_offset() == spec.page_size_padded // elem_size
    # Within one block the (unpadded) content stays compact.
    assert kv_cache[0].is_contiguous()


@pytest.mark.parametrize(
    (
        "kernel_block_sizes",
        "expected_num_blocks",
        "expected_num_states",
    ),
    [
        (None, 4, 64),
        ([256], 4, 64),
        ([64], 16, 16),
    ],
)
def test_allocate_compressed_mla_cache(
    kernel_block_sizes: list[int] | None,
    expected_num_blocks: int,
    expected_num_states: int,
):
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    num_pages = 4
    config = KVCacheConfig(
        num_blocks=num_pages,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_pages * spec.page_size_bytes,
                layers=["layer.0"],
                layer_stride=num_pages * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
    )

    caches = allocate_kv_cache(
        config, torch.device("cpu"), KVCacheLayout.LBHNC, kernel_block_sizes
    )

    assert caches["layer.0"].shape == (expected_num_blocks, 1, expected_num_states, 128)


@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_copy_kv_cache_blocks_shared_storage(layout: KVCacheLayout):
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(num_blocks):
            cache[block_idx].fill_(10 * layer_idx + block_idx)

    expected = [[cache[i].clone() for i in range(num_blocks)] for cache in caches]
    copies = [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)]

    copy_kv_cache_blocks_inplace(caches, num_blocks, copies)

    for layer_idx, cache in enumerate(caches):
        torch.testing.assert_close(cache[2], expected[layer_idx][0])
        torch.testing.assert_close(cache[1], expected[layer_idx][1])


def test_fixed_block_stride_propagates_outward_in_lhbnc():
    num_blocks = 3
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
    )
    natural = compute_layout_strides(spec, num_blocks, num_layers, KVCacheLayout.LHBNC)
    block_stride = natural[1] + 8

    strides = compute_layout_strides(
        spec,
        num_blocks,
        num_layers,
        KVCacheLayout.LHBNC,
        fixed_strides=(None, block_stride, None, None, None),
    )

    assert strides[1] == block_stride
    assert strides[2] == block_stride * num_blocks
    assert strides[0] == strides[2] * spec.num_heads


def test_copy_kv_cache_blocks_separate_head_groups():
    # LHBNC stores each head group separately, so a block's bytes are scattered
    # across L*H regions.
    layout = KVCacheLayout.LHBNC
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
        num_head_slots=2,
        state_content_bytes=2 * 2 * 4,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(num_blocks):
            for head_idx in range(cache.shape[1]):
                cache[block_idx, head_idx].fill_(
                    100 * layer_idx + 10 * head_idx + block_idx
                )

    expected = [[cache[i].clone() for i in range(num_blocks)] for cache in caches]
    copy_kv_cache_blocks_inplace(
        caches,
        num_blocks,
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
    )

    for layer_idx, cache in enumerate(caches):
        torch.testing.assert_close(cache[2], expected[layer_idx][0])
        torch.testing.assert_close(cache[1], expected[layer_idx][1])


@pytest.mark.parametrize(
    "layout,num_layers",
    [
        (KVCacheLayout.LBHNC, 2),
        # Splitting needs a manager block to be one dense page, which a
        # block-outermost layout only gives when the block holds one layer.
        (KVCacheLayout.BLHNC, 1),
    ],
)
def test_copy_kv_cache_blocks_with_virtual_block_splitting(
    layout: KVCacheLayout, num_layers: int
):
    num_blocks = 4
    physical_per_logical = 2
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(
        raw,
        spec,
        num_blocks,
        num_layers,
        layout,
        kernel_block_size=spec.block_size // physical_per_logical,
    )

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(cache.shape[0]):
            cache[block_idx].fill_(100 * layer_idx + block_idx)
    expected = [[cache[i].clone() for i in range(cache.shape[0])] for cache in caches]

    copy_kv_cache_blocks_inplace(
        caches,
        num_blocks,
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
    )

    dst_start = 2 * physical_per_logical
    for layer_idx, cache in enumerate(caches):
        for physical_idx in range(physical_per_logical):
            torch.testing.assert_close(
                cache[dst_start + physical_idx], expected[layer_idx][physical_idx]
            )


@pytest.mark.parametrize("num_kv_heads", [1, 2])
@pytest.mark.parametrize(
    "supported,requested,expected_error",
    [
        (["BLNHC", "BLHNC"], None, None),
        (["BLHNC"], None, "requires BLNHC"),
        (["BLNHC", "BLHNC"], "BLHNC", "valid layouts: .*BLNHC"),
    ],
)
def test_hisparse_full_kv_resolves_contiguous_token_rows(
    monkeypatch, num_kv_heads, supported, requested, expected_error
):
    """The MLA connector preference cannot override full-K/V row geometry."""
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    if requested is not None:
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", requested)
    from tests.v1.core.test_kv_cache_planning import _hisparse_planning_config

    config = _hisparse_planning_config()
    # This core entrypoint resolves an unset layout; a pre-resolved value wins.
    config.cache_config.kv_cache_layout = None
    spec = SparseFullAttentionSpec(
        block_size=16,
        num_kv_heads=num_kv_heads,
        total_num_kv_heads=num_kv_heads,
        head_size=64,
        dtype=torch.bfloat16,
        top_k=32,
    )
    specs = [
        spec,
        HiSparseResidentSpec(block_size=16, page_size=spec.page_size_bytes),
        HiSparseHotSpec(
            block_size=16, page_size=spec.page_size_bytes, blocks_per_request=4
        ),
    ]

    if expected_error is not None:
        with pytest.raises(ValueError, match=expected_error):
            resolve_kv_cache_layout(config, [supported], iter(specs))
        assert config.cache_config.kv_cache_layout is None
        return

    layout = resolve_kv_cache_layout(config, [supported], iter(specs))

    assert layout is KVCacheLayout.BLNHC
    assert config.cache_config.get_resolved_kv_cache_layout() is layout


@pytest.mark.parametrize(
    "spec_type,hisparse_enabled",
    [
        (FullAttentionSpec, False),
        (SparseFullAttentionSpec, False),
        (MLAAttentionSpec, True),
    ],
)
def test_hisparse_full_kv_layout_preserves_other_cache_preferences(
    monkeypatch, spec_type, hisparse_enabled
):
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    from tests.v1.core.test_kv_cache_planning import _hisparse_planning_config

    config = _hisparse_planning_config()
    # This core entrypoint resolves an unset layout; a pre-resolved value wins.
    config.cache_config.kv_cache_layout = None
    config.attention_config.hisparse_config = (
        HiSparseConfig() if hisparse_enabled else None
    )
    config.kv_transfer_config = (
        KVTransferConfig(kv_connector="HiSparseConnector", kv_role="kv_both")
        if hisparse_enabled
        else None
    )
    extra = (
        {"total_num_kv_heads": 1, "top_k": 32}
        if spec_type is SparseFullAttentionSpec
        else {}
    )
    spec = spec_type(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16, **extra
    )
    supported = ["BLNHC", "BLHNC"] if hisparse_enabled else ["BLHNC", "BLNHC"]

    assert resolve_kv_cache_layout(config, [supported], [spec]) is KVCacheLayout.BLHNC


def test_allocate_hisparse_kv_caches_host_pool_and_view_less_specs():
    """Host tensors get their own backing; view-less specs keep the raw one."""
    spec = FullAttentionSpec(
        block_size=2, num_kv_heads=1, head_size=4, dtype=torch.float32
    )
    page = spec.page_size_bytes
    resident_spec = HiSparseResidentSpec(block_size=2, page_size=page)
    device_size = 4 * page
    config = KVCacheConfig(
        num_blocks=4,
        hisparse_host_num_blocks=3,
        kv_cache_tensors=[
            KVCacheTensor(
                size=3 * page,
                layers=["source"],
                layer_stride=3 * page,
                block_stride=page,
                host_resident=True,
            ),
            KVCacheTensor(
                size=device_size,
                layers=["indexer"],
                layer_stride=device_size,
                block_stride=page,
            ),
            KVCacheTensor(
                size=device_size,
                layers=["resident"],
                layer_stride=device_size,
                block_stride=page,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["source"], spec, host_resident=True),
            KVCacheGroupSpec(["indexer"], spec),
            KVCacheGroupSpec(["resident"], resident_spec),
        ],
    )
    host_buffers: list[torch.Tensor] = []

    def host_allocator(size: int) -> torch.Tensor:
        host_buffers.append(torch.zeros(size, dtype=torch.int8))
        return host_buffers[-1]

    caches = allocate_hisparse_kv_caches(
        config,
        torch.device("cpu"),
        KVCacheLayout.LBHNC,
        [2, 2, 2],
        SimpleNamespace(allocate=host_allocator),
    )
    assert len(config.kv_cache_tensors) == 3

    assert [buf.numel() for buf in host_buffers] == [3 * page]
    assert caches["source"].shape[0] == 3
    assert (
        caches["source"].untyped_storage().data_ptr()
        == host_buffers[0].untyped_storage().data_ptr()
    )
    assert caches["indexer"].shape[0] == 4
    backing = caches["resident"]
    assert backing.dtype == torch.int8 and backing.numel() >= device_size
    assert (
        backing.untyped_storage().data_ptr()
        == caches["indexer"].untyped_storage().data_ptr()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA cache binding")
def test_bind_hisparse_full_kv_preserves_token_head_and_kv_rows(dist_init):
    """Real binding preserves every token/head/K/V row and host storage owner."""
    from tests.v1.core.test_kv_cache_planning import _hisparse_planning_config
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    from vllm.v1.hisparse.runtime import (
        HiSparseHostPool,
        ResolvedHiSparseConfig,
        release_pinned_state,
    )
    from vllm.v1.worker.gpu.block_table import BlockTables

    names = ["layer.0", "layer.1"]
    resident_names = [f"{name}.hisparse_resident" for name in names]
    hot_names = [f"{name}.hisparse_hot" for name in names]
    heads, dim, block_size = 2, 64, 2
    row_width = heads * 2 * dim
    page_bytes = block_size * row_width * torch.bfloat16.itemsize
    spec = SparseFullAttentionSpec(
        block_size=block_size,
        num_kv_heads=heads,
        total_num_kv_heads=2 * heads,
        head_size=dim,
        dtype=torch.bfloat16,
        top_k=2,
    )
    source_spec = UniformTypeKVCacheSpecs.from_specs(dict.fromkeys(names, spec))
    assert source_spec is not None
    cache_config = KVCacheConfig(
        num_blocks=4,
        hisparse_host_num_blocks=3,
        kv_cache_tensors=[
            KVCacheTensor(
                size=6 * page_bytes,
                layers=names,
                layer_stride=3 * page_bytes,
                block_stride=page_bytes,
                host_resident=True,
            ),
            KVCacheTensor(
                size=8 * page_bytes,
                layers=resident_names,
                layer_stride=page_bytes,
                block_stride=2 * page_bytes,
            ),
            KVCacheTensor(
                size=8 * page_bytes,
                layers=hot_names,
                layer_stride=page_bytes,
                block_stride=2 * page_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(names, source_spec, host_resident=True),
            KVCacheGroupSpec(
                resident_names, HiSparseResidentSpec(block_size=2, page_size=page_bytes)
            ),
            KVCacheGroupSpec(
                hot_names,
                HiSparseHotSpec(
                    block_size=2, page_size=page_bytes, blocks_per_request=2
                ),
            ),
        ],
    )
    config = _hisparse_planning_config()
    config.attention_config.flash_attn_version = 2
    device = torch.device("cuda")
    host_pool = HiSparseHostPool(config, cache_config)
    caches = allocate_hisparse_kv_caches(
        cache_config, device, KVCacheLayout.BLNHC, [2, 2, 2], host_pool
    )
    block_tables = BlockTables([2, 2, 2], 1, 2, [2, 2, 2], device)
    for table in block_tables.input_block_tables:
        table.copy_(torch.tensor([[1, 2]], dtype=torch.int32, device=device))
    forward_context, expected_rows, runtimes = {}, [], []
    try:
        with set_current_vllm_config(config), torch.device(device):
            for index, name in enumerate(names):
                tagged = (
                    torch.arange(6 * row_width, dtype=torch.float32).view(
                        3, 2, heads, 2 * dim
                    )
                    + index * 64
                ).to(torch.bfloat16)
                caches[name].copy_(tagged.transpose(1, 2))
                expected_rows.append(tagged.view(6, row_width).cpu())
                layer = Attention(
                    heads,
                    dim,
                    dim**-0.5,
                    num_kv_heads=heads,
                    cache_config=config.cache_config,
                    prefix=name,
                    attn_backend=FlashAttentionBackend,
                )
                runtime = HiSparseRuntime(
                    ResolvedHiSparseConfig(2, 4, 2),
                    1,
                    row_width,
                    torch.bfloat16,
                    device,
                )
                runtimes.append(runtime)
                layer.hisparse_cache = HiSparseCacheHandle(runtime)
                layer.is_draft_layer = index == 1
                forward_context[name] = layer
        handles = attn_utils_module.bind_hisparse_kv_caches(
            forward_context=forward_context,
            kv_cache_config=cache_config,
            kv_caches=caches,
            block_tables=block_tables,
            host_pool=host_pool,
        )
        assert len(handles) == 2
        residency = handles[0].residency
        assert isinstance(residency, torch.Tensor)
        assert residency.shape == (1, 1, 2)
        assert torch.count_nonzero(residency).item() == 0
        for index, (name, handle) in enumerate(zip(names, handles, strict=True)):
            assert handle.draft_layer == (index == 1)
            assert caches[name].shape == (3, heads, 2, 2 * dim)
            assert handle.runtime.host_cache.shape == (6, row_width)
            assert handle.runtime.host_cache.data_ptr() == caches[name].data_ptr()
            torch.testing.assert_close(handle.runtime.host_cache, expected_rows[index])
            assert handle.view is not None
            assert handle.view.cache.shape == (4, 2, row_width)
            assert handle.view.cache.stride() == (2 * page_bytes // 2, row_width, 1)
            assert handle.runtime.hot.cache.stride() == (
                2 * page_bytes // 2,
                row_width,
                1,
            )
            assert handle.source_block_table is block_tables.input_block_tables[0]
            assert handle.residency is residency
            assert isinstance(residency, torch.Tensor)
            assert handle.block_table is not None
            assert handle.block_table.data_ptr() == residency[:, 0].data_ptr()
        residency[0, 0, 0] = 3
        for handle in handles:
            assert handle.block_table is not None
            assert handle.block_table.tolist() == [[3, 0]]
        assert block_tables.input_block_tables[0].tolist() == [[1, 2]]
    finally:
        release_pinned_state(runtimes, [host_pool.registered])


class _TableBuilder:
    requires_block_table_width = False
    supports_update_block_table = False

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        return common_attn_metadata.block_table_tensor


def _init_mamba_reuse_groups():
    """Discover real layers/backends and construct metadata without forwards."""
    import sys
    from dataclasses import replace

    from transformers import Lfm2Config, Qwen4ExpTextConfig

    from tests.v1.core.test_kv_cache_planning import _hisparse_planning_config
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.mamba.mamba_mixer import MambaMixer
    from vllm.model_executor.layers.mamba.short_conv import ShortConv
    from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPLELayer
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

    config = _hisparse_planning_config()
    config.attention_config.hisparse_config = None
    config.kv_transfer_config = None
    config.scheduler_config.max_num_seqs = 4
    config.scheduler_config.max_num_batched_tokens = 16
    config.cache_config.mamba_block_size = 16
    config.cache_config.mamba_cache_mode = "none"
    config.parallel_config.enable_dbo = True
    config.attention_config.flash_attn_version = 2
    device = torch.device("cuda")
    conv = Lfm2Config(conv_L_cache=2, conv_bias=False)
    ple = Qwen4ExpTextConfig(
        hidden_size=8,
        hc_count=2,
        ple_embed_dim=8,
        ple_conv_kernel_size=2,
        ngram_size=2,
        heads_per_ngram=1,
        ngram_vocab_size_base=5,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=2,
        vocab_size=32,
        eos_token_id=0,
    )
    with set_current_vllm_config(config), torch.device(device):
        layers = [
            ShortConv(
                conv,
                8,
                i,
                config.model_config,
                config.cache_config,
                prefix=f"layer.{i}",
            )
            for i in range(3)
        ]
        layers.append(
            MambaMixer(
                8,
                2,
                2,
                8,
                1,
                False,
                False,
                False,
                model_config=config.model_config,
                cache_config=config.cache_config,
                prefix="layer.3",
            )
        )
        layers.append(Qwen4ExpPLELayer(ple, config, prefix="layer.4"))
        layers.append(
            Attention(
                1,
                64,
                0.125,
                num_kv_heads=1,
                cache_config=config.cache_config,
                prefix="layer.5",
                attn_backend=FlashAttentionBackend,
            )
        )
    # Metadata tests never allocate these cache pages. Python int hash's documented
    # modulus supplies distinct real specs with identical hashes, without replacing
    # MambaSpec.__hash__ or any metadata builder.
    layer_spec = layers[0].get_kv_cache_spec(config)
    assert isinstance(layer_spec, MambaSpec)
    spec = replace(layer_spec, page_size_padded=2 * sys.hash_info.modulus)
    other = replace(spec, page_size_padded=sys.hash_info.modulus)
    assert spec != other and hash(spec) == hash(other)
    specs = [
        spec,
        replace(spec),
        other,
        spec,
        spec,
        layers[5].get_kv_cache_spec(config),
    ]
    cache = KVCacheConfig(
        8,
        [],
        [KVCacheGroupSpec([f"layer.{i}"], value) for i, value in enumerate(specs)],
    )
    with set_current_vllm_config(config):
        groups, _, _ = attn_utils.init_attn_backend(cache, config, device)
    return groups, cache, config, device


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="real CUDA layer initialization"
)
@pytest.mark.parametrize("manual", [None, 0, 1])
@pytest.mark.parametrize("changed", [None, "class", "spec", "optout"])
@pytest.mark.parametrize("ubatch", [0, 1])
@pytest.mark.parametrize("capture", [False, True])
def test_mamba_metadata_reuse_preserves_value_and_builder_boundaries(
    manual, changed, ubatch, capture, dist_init
):
    """Real prepared/manual groups preserve spec, backend and opt-out boundaries."""
    from contextlib import ExitStack
    from unittest.mock import patch

    from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionMetadataBuilder
    from vllm.v1.attention.backends.short_conv_attn import (
        PleShortConvAttentionMetadataBuilder,
    )

    groups, cache, config, device = _init_mamba_reuse_groups()
    if manual is not None:
        group = groups[manual][0]
        groups[manual][0] = AttentionGroup(
            group.backend,
            group.layer_names,
            group.kv_cache_spec,
            manual,
            metadata_builders=group.metadata_builders,
        )
    group = groups[1][0]
    if changed == "class":
        group.metadata_builders[ubatch] = Mamba1AttentionMetadataBuilder(
            group.kv_cache_spec, group.layer_names, config, device
        )
    elif changed == "spec":
        group.kv_cache_spec = groups[2][0].kv_cache_spec
    elif changed == "optout":
        group.metadata_builders[ubatch] = PleShortConvAttentionMetadataBuilder(
            group.kv_cache_spec, group.layer_names, config, device
        )
    builders = [g[0].get_metadata_builder(ubatch) for g in groups]
    with ExitStack() as observers:
        calls = [
            observers.enter_context(patch.object(b, "build", wraps=b.build))
            for b in builders
        ]
        updates = [
            observers.enter_context(
                patch.object(b, "update_block_table", wraps=b.update_block_table)
            )
            for b in builders
        ]
        for step in range(2):
            prior = sum(call.call_count for call in calls)
            positions = torch.tensor([0, 1], dtype=torch.int32, device=device)
            actual = attn_utils.build_attn_metadata(
                attn_groups=groups,
                num_reqs=1,
                num_tokens=1,
                query_start_loc_gpu=positions,
                query_start_loc_cpu=positions.cpu(),
                max_query_len=1,
                seq_lens=torch.tensor([40], device=device),
                seq_lens_cpu_upper_bound=torch.tensor([40], dtype=torch.int32),
                is_prefilling=torch.zeros(1, dtype=torch.bool),
                max_seq_len=40,
                block_tables=[
                    torch.tensor([[i + 100 * step]], dtype=torch.int32, device=device)
                    for i in range(6)
                ],
                slot_mappings=torch.zeros(6, 1, dtype=torch.int64, device=device),
                kv_cache_config=cache,
                for_cudagraph_capture=capture,
                ubatch_idx=ubatch,
            )
            metadata = [actual[f"layer.{i}"] for i in range(6)]
            values = [m.state_indices_tensor_d.item() for m in metadata[:5]]
            values.append(metadata[5].block_table.item())
            assert values == [i + 100 * step for i in range(6)]
            expected_batches = 6 if changed == "optout" else 5
            assert sum(call.call_count for call in calls) - prior == expected_batches
            reused = 3 if changed == "class" else 2 if changed == "spec" else 1
            if changed != "optout":
                assert updates[reused].call_count == step + 1
            else:
                assert all(call.call_count == 0 for call in updates)


_VLLM_CONFIG = SimpleNamespace(
    model_config=SimpleNamespace(max_model_len=8192),
    scheduler_config=SimpleNamespace(max_num_seqs=4),
    parallel_config=SimpleNamespace(decode_context_parallel_size=1),
)


def _packed_group(tokens_per_state, row_bytes, supported_sizes):
    spec = MLAAttentionSpec(
        block_size=1024,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        state_content_bytes=row_bytes,
        tokens_per_state=tokens_per_state,
    )
    backend = SimpleNamespace(
        get_builder_cls=lambda: _TableBuilder,
        get_supported_kernel_block_sizes=lambda _=None: supported_sizes,
        get_name=lambda: "FAKE",
        customize_spec=lambda spec: spec,
    )
    return AttentionGroup(backend, ["layer.0"], spec, 0)


def test_packed_compressed_group_gets_kernel_block_table():
    """A packed indexer cache whose kernel needs 64-state pages gets a table and
    a bound view in kernel blocks that address the same packed-block rows."""
    row, stride_rows = 132, 5 * 64
    group = _packed_group(4, row, [128, 256])
    group.create_metadata_builders(_VLLM_CONFIG, "cpu", kernel_block_size=1024)
    assert group.get_metadata_builder().kv_cache_spec.block_size == 256

    backing = torch.arange(3 * stride_rows).repeat_interleave(row)
    manager = backing.view(3, 1, stride_rows, row)[:, :, :256]
    pages = map_kv_caches_to_kernel_blocks({"layer.0": manager}, [group])["layer.0"]
    metadata = create_common_attn_metadata(BatchSpec([2048], [1]), 1024, "cpu")
    metadata.block_table_tensor = torch.tensor([[0, 2]], dtype=torch.int32)
    table = group.build_metadata(metadata)
    assert table.tolist() == [[0, 1, 2, 3, 10, 11, 12, 13]]
    assert torch.equal(pages[table[0, 4:], 0].flatten(0, 1), manager[2, 0])


def test_split_group_maps_manager_block_table():
    """Groups split in a dense layout keep manager block tables; builders get
    b * blocks_per_kv_block + j against the view allocated in kernel blocks."""
    group = _packed_group(4, 132, [128, 256])
    group.create_metadata_builders(_VLLM_CONFIG, "cpu", kernel_block_size=256)
    kernel_view = torch.zeros(12, 1, 64, 132)
    bound = map_kv_caches_to_kernel_blocks({"layer.0": kernel_view}, [group])
    assert bound["layer.0"] is kernel_view

    metadata = create_common_attn_metadata(BatchSpec([2048], [1]), 1024, "cpu")
    metadata.block_table_tensor = torch.tensor([[0, 2]], dtype=torch.int32)
    table = group.build_metadata(metadata)
    assert table.tolist() == [[0, 1, 2, 3, 8, 9, 10, 11]]


@pytest.mark.parametrize(
    ("block_size", "page_states"), [(1024, 64), (384, 32), (256, 0)]
)
def test_split_blocks_align_to_kernel_blocks(block_size, page_states):
    """Backends taking only fixed kernel blocks align packed blocks to the kernel
    block the final block size is split into, and not at all when unsplit."""
    group = _packed_group(4, 132, [128, 256])
    spec = customize_attention_spec(group.backend, group.kv_cache_spec)
    spec = spec.copy_with_new_block_size(block_size)
    assert spec.get_block_stride_alignment() == max(page_states * 132, 1)


def test_packed_token_cache_cannot_be_split():
    group = _packed_group(1, 1152, [64])
    group.create_metadata_builders(_VLLM_CONFIG, "cpu", kernel_block_size=1024)
    packed = torch.zeros(3, 1, 2048, 1152, dtype=torch.uint8)[:, :, :1024]
    with pytest.raises(ValueError, match="cannot be split"):
        map_kv_caches_to_kernel_blocks({"layer.0": packed}, [group])
