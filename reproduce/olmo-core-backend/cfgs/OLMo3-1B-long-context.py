"""
OLMo 3 1B stage-3 long-context extension configuration.

This is a 1B adaptation of the OLMo 3 7B long-context recipe in
`src/scripts/official/OLMo3/OLMo-3-1025-7B-long-context.py`. OLMo 3 does not publish an
officially tuned 1B long-context recipe.

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

Other conflicts: flash_3 has no CP, use flash_2 for CP;
TP + EP is forbidden. Multi-stage PP + tied embeddings is forbidden.
Stage 2/3 optimizer states must have the same optimizer type unless loading is disabled.

Examples: world_size=64 (GPUs), PP=1 (off), TP=1 (off), heads=16
B=2^22 (tokens), M=L=65,536 (tokens)
| Optim | DP layout               | CP | Muon mesh | Result                        |
|-------|-------------------------|----|-----------|-------------------------------|
| AdamW | HSDP H_rep=16,H_shard=1 | 4  | -         | valid                         |
| AdamW | HSDP H_rep=8,H_shard=1  | 8  | -         | valid                         |
| Muon  | HSDP H_rep=8,H_shard=8  | 1  | 8         | valid:16(heads)%8(mesh)=0     |
| Muon  | FSDP DP=64              | 1  | 64        | invalid:16(heads)%64(mesh)!=0 |
| Muon  | FSDP DP=16              | 4  | DP*CP=64  | invalid:16(heads)%64(mesh)!=0 |
"""

import argparse
from typing import List

from _olmo3_1b import build_common_config, build_optim_config, get_olmo3_1b_cli_parser

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import LinearWithWarmup
from olmo_core.script_utils import ExperimentConfig, main
from olmo_core.train.common import LoadStrategy
from olmo_core.train.train_module import (
    TransformerContextParallelConfig,  # noqa: F401 - used by the optional cp_config below
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

DEFAULT_SEQUENCE_LENGTH = 65536
GLOBAL_BATCH_SIZE = 2**22  # 4M tokens
# MAX_TOKENS = 50_000_000_000  # 50B
# Muon retains the 1B recipe; AdamW follows the official stage-3 schedule.
MUON_LR = 5e-4
ADAM_LR = 5e-4
SEED = 4123


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    """Build stage 3 from its required components and the shared trainer."""
    # Long context changes the model, dataset, loader, and train module as whole
    # units, so this stage does not mutate the stage-1 versions of those components.
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH
    tokenizer_config = TokenizerConfig.dolma2()

    model = TransformerConfig.olmo3_1B(
        vocab_size=tokenizer_config.padded_vocab_size(),  # pad to a multiple of 128
        attn_backend=AttentionBackendName.flash_3,
    ).with_rope_scaling(
        YaRNRoPEScalingConfig(
            factor=8,
            beta_fast=32,
            beta_slow=1,
            old_context_len=8192,
        )
    )

    dataset = NumpyPackedFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_longmino_mix_0625,
        mix_base_dir=opts.data_root,
        work_dir=opts.work_dir,
        tokenizer=tokenizer_config,
        sequence_length=sequence_length,
        generate_doc_lengths=True,  # enables intra-document masking
        source_group_size=8,
        source_permutation_seed=123,
    )

    data_loader = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=SEED,
        num_workers=8,
        prefetch_factor=4,
    )

    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=sequence_length,
        max_sequence_length=sequence_length,
        optim=build_optim_config(
            opts.optim,
            muon_lr=MUON_LR,
            adam_lr=ADAM_LR,
        ),
        scheduler=LinearWithWarmup(warmup=200, alpha_f=0.0),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        # cp_config=TransformerContextParallelConfig.llama3(degree=4, head_stride=4),
        ac_config=None,
        float8_config=None,
        # float8_config=Float8Config(enabled=True, ao=AOFloat8LinearConfig.recommended()),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    # Only the trainer and its common callbacks are inherited from stage 1.
    config = build_common_config(
        opts,
        model=model,
        dataset=dataset,
        data_loader=data_loader,
        train_module=train_module,
    )

    config.trainer.load_strategy = LoadStrategy.always
    # script_utils.main probes save_folder before Trainer.fit() and uses this value, so
    # require trainer state for a same-stage resume. The launcher supplies the parent
    # stage through ExperimentConfig.load_path, which explicitly skips trainer state.
    config.trainer.load_trainer_state = True
    config.trainer.load_optim_state = True

    config.init_seed = SEED
    return config.merge(overrides)


if __name__ == "__main__":
    main(build_config, parser=get_olmo3_1b_cli_parser())
