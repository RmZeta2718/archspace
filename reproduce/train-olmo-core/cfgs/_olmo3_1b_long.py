"""
Shared configuration for the OLMo 3 1B long-context and SFT stages.

Parallelism boundaries
----------------------
DP=data-parallel world size; PP/CP/TP/EP are their degrees.
H_rep=HSDP replicas, H_shard=HSDP shard degree, L=seqlen, M=microbatch tokens,
B=global batch tokens; heads=16; n_layers=16.

Mesh:  world_size = PP*CP*TP*DP; world_size % (PP*CP*TP) = 0
HSDP:  DP = H_rep*H_shard; DP % H_shard = 0
Batch: M % L = 0; B % (M*DP) = 0; grad_accum = B/(M*DP)
CP:    local_L = L/CP; exact split requires L % CP = 0
Ulysses CP: q_heads % CP = kv_heads % CP = 0
TP:    tensor_dim % TP = 0 for every sharded dimension
PP:    world_size % PP = 0; num_stages % PP = 0; num_stages <= n_layers
EP:    MoE and HSDP only; EP = H_shard; TP = 1 (off)

Optimizer / parallelism matrix:
| Mode         | AdamW           | Muon                                 |
|--------------|-----------------|--------------------------------------|
| FSDP         | yes             | yes: heads % (DP*CP) = 0             |
| HSDP, CP off | yes             | yes: heads % H_shard = 0             |
| HSDP + CP    | yes             | no: dp_shard is flattened into dp_cp |
| TP           | yes             | no: hard error                       |
| PP           | yes (beta)      | beta; changes DP                     |
| EP           | MoE + HSDP only | no: flattened/3D expert parameters   |

Other conflicts: flash_3 has no CP, use flash_2 for CP; Muon also has no CP.
TP + EP is forbidden. Multi-stage PP + tied embeddings is forbidden.
Adjacent stages must keep the same optimizer type and parameter groups while optimizer-state
loading is enabled.

Stage-3 examples: world_size=64 (GPUs), PP=1 (off), TP=1 (off), heads=16
B=2^21 (tokens), M=L=32,768 (tokens)
| Optim | DP layout               | CP | Muon mesh | Result                        |
|-------|-------------------------|----|-----------|-------------------------------|
| AdamW | HSDP H_rep=16,H_shard=1 | 4  | -         | valid                         |
| AdamW | HSDP H_rep=8,H_shard=1  | 8  | -         | valid                         |
| Muon  | HSDP H_rep=8,H_shard=8  | 1  | 8         | valid:16(heads)%8(mesh)=0     |
| Muon  | FSDP DP=64              | 1  | 64        | invalid:16(heads)%64(mesh)!=0 |
| Muon  | FSDP DP=16              | 4  | DP*CP=64  | invalid:16(heads)%64(mesh)!=0 |
"""

import argparse

from _olmo3_1b_base import (
    build_common_config,
    build_optim_config,
    configure_stage_continuation,
    num_workers,
    prefetch_factor,
    seed,
)

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.types import LongDocStrategy
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import (  # noqa: F401 - optional commented configuration below
    AOFloat8LinearConfig,
    Float8Config,
)
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import LinearWithWarmup
from olmo_core.script_utils import ExperimentConfig
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,  # noqa: F401 - optional commented config below
    TransformerActivationCheckpointingMode,  # noqa: F401 - optional commented config below
    TransformerContextParallelConfig,  # noqa: F401 - optional commented config below
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

# The long-context model/YaRN pattern, packed-dataset structure, and linear scheduler are adapted from:
# src/scripts/official/OLMo3/OLMo-3-1025-7B-long-context.py.

source_group_size = 8


def build_long_context_model(tokenizer: TokenizerConfig) -> TransformerConfig:
    """Build the complete checkpoint-compatible model used by stages 3-5."""
    return TransformerConfig.olmo3_1B(
        vocab_size=tokenizer.padded_vocab_size(),  # pad to a multiple of 128
        attn_backend=AttentionBackendName.flash_3,
    ).with_rope_scaling(
        YaRNRoPEScalingConfig(
            factor=8,
            beta_fast=32,
            beta_slow=1,
            old_context_len=4096,
        )
    )


def build_long_context_config(
    opts: argparse.Namespace,
    *,
    lr: float,
    sequence_length: int,
    rank_microbatch_size: int,
    global_batch_size: int,
) -> ExperimentConfig:
    """Build the long-context baseline inherited by both SFT phases."""
    sequence_length = opts.sequence_length or sequence_length
    tokenizer = TokenizerConfig.dolma2()

    model = build_long_context_model(tokenizer)

    # Longmino packed data records document lengths for isolated attention and jointly packs
    # consecutive source files in resolved data-mix order, but has no label-mask sidecar.
    dataset = NumpyPackedFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_longmino_mix_0625,
        tokenizer=tokenizer,
        mix_base_dir=opts.data_root,
        sequence_length=sequence_length,
        generate_doc_lengths=True,  # enables intra-document masking
        long_doc_strategy=LongDocStrategy.truncate,
        source_group_size=source_group_size,
        work_dir=opts.work_dir,
    )

    data_loader = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size,
        seed=seed,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    # Model, packed dataset, loader, and train module all differ materially from stage 2, so this
    # baseline constructs those configs in full instead of overwriting short-context members.
    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=sequence_length,
        optim=build_optim_config(opts.optim, lr=lr),
        scheduler=LinearWithWarmup(warmup=200, alpha_f=0.0),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        # The API alternatives below affect stages 3-5; use CLI overrides for one stage only.
        # Optional shared context-parallel profiles. CP cannot be used with FlashAttention 3:
        # switch the model backend to FlashAttention 2 and select Adam pipeline-wide first.
        # Common CLI: `--model.block.sequence_mixer.backend=flash_2 --optim=adam`.
        # Stage-3 CLI: `'--train_module.cp_config={degree: 4, ring: {load_balancer: llama3, head_stride: 4}}'`.
        # cp_config=TransformerContextParallelConfig.llama3(degree=4, head_stride=4),
        # SFT CLI: `'--train_module.cp_config={degree: 2, ring: {load_balancer: llama3, head_stride: 4}}'`.
        # cp_config=TransformerContextParallelConfig.llama3(degree=2, head_stride=4),
        cp_config=None,
        # Optional activation-checkpointing profiles.
        # CLI: `'--train_module.ac_config={mode: budget, activation_memory_budget: 0.7}'`.
        # ac_config=TransformerActivationCheckpointingConfig(
        #     mode=TransformerActivationCheckpointingMode.budget,
        #     activation_memory_budget=0.7,
        # )
        # CLI: `'--train_module.ac_config={mode: selected_modules, modules: ["blocks.*.feed_forward"]}'`.
        # ac_config=TransformerActivationCheckpointingConfig(
        #     mode=TransformerActivationCheckpointingMode.selected_modules,
        #     modules=["blocks.*.feed_forward"],
        # )
        ac_config=None,
        # Published 7B float8 profile; the local 1B baseline keeps float8 disabled.
        # CLI: `'--train_module.float8_config={enabled: true, ao: {enable_fsdp_float8_all_gather: true, force_recompute_fp8_weight_in_bwd: true, round_scales_to_power_of_2: true}}'`.
        # float8_config=Float8Config(
        #     enabled=True,
        #     ao=AOFloat8LinearConfig.recommended(),
        # )
        float8_config=None,
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    config = build_common_config(
        opts,
        model=model,
        dataset=dataset,
        data_loader=data_loader,
        train_module=train_module,
    )
    configure_stage_continuation(config.trainer)
    return config
