"""Shared configuration for the OLMo 3 1B training stages."""

import argparse

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.eval.task_groups import FAST_TASKS
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import (
    CosWithWarmup,
    MuonConfig,
    OptimConfig,
    OptimGroupOverride,
    SkipStepAdamWConfig,
)
from olmo_core.script_utils import ExperimentConfig, get_cli_parser
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    LMEvaluatorCallbackConfig,
    MonkeyPatcherCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

DEFAULT_SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE = 2**21  # 2M tokens
SEED = 34521
EVAL_LM_STEPS = 500  # 500 steps (~1B token) for 150B data, 2500 steps (~5B token) for 6T data.
EVAL_DOWN_STEPS = 12500  # 12.5K steps (25B tokens) for 150B data
# Keep the current Muon recipe and the official OLMo 3 AdamW recipe independent.
MUON_LR = 1e-3
ADAM_LR = 1e-3


def get_olmo3_1b_cli_parser() -> argparse.ArgumentParser:
    """Build the CLI parser shared by the OLMo 3 1B stages."""
    parser = get_cli_parser()
    parser.add_argument(
        "--optim",
        choices=("adam", "muon"),
        default="muon",
        help="Optimizer recipe to use; adam selects SkipStep AdamW (default: muon).",
    )
    return parser


def build_optim_config(
    name: str,
    *,
    muon_lr: float,
    adam_lr: float,
) -> OptimConfig:
    """Build the selected optimizer with its stage-specific learning rate."""
    # Equivalent whole-object CLI override; define `lr` in the shell first:
    # "--train_module.optim={type: muon, lr: ${lr}, weight_decay: 0.033, betas: [0.9, 0.95]}"
    if name == "muon":
        return MuonConfig(
            lr=muon_lr,
            weight_decay=0.033,
            betas=(0.9, 0.95),
        )
    # Equivalent whole-object CLI override; define `lr` in the shell first:
    # "--train_module.optim={type: skip_step_adamw, lr: ${lr}, weight_decay: 0.033, betas: [0.9, 0.95], group_overrides: [{params: [embeddings.weight], opts: {weight_decay: 0.0}}]}"
    if name == "adam":
        # Match the official OLMo 3 AdamW recipe, including no decay on embeddings.
        return SkipStepAdamWConfig(
            lr=adam_lr,
            weight_decay=0.033,
            betas=(0.9, 0.95),
            group_overrides=[OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0})],
        )
    raise ValueError(f"Unknown optimizer '{name}'")


def build_common_config(
    opts: argparse.Namespace,
    *,
    model: TransformerConfig,
    dataset: NumpyDatasetConfig,
    data_loader: NumpyDataLoaderConfig,
    train_module: TransformerTrainModuleConfig,
) -> ExperimentConfig:
    """Build an experiment from required stage components and the shared trainer."""
    # Temporary checkpoint approximately every 1B tokens.
    ephemeral_save_interval = round(2**30 / data_loader.global_batch_size / 10) * 10

    trainer = (
        TrainerConfig(
            save_folder=opts.save_folder,
            work_dir=opts.work_dir,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.epochs(1),
        )
        .with_callback("monkey_patcher", MonkeyPatcherCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=None,  # Only save the final ckpt
                ephemeral_save_interval=ephemeral_save_interval,
                max_checkpoints=1,
                # pre_train_checkpoint=False,
                # save_async=False,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=opts.name,
                cancel_check_interval=10,
                enabled=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.name,
                cancel_check_interval=10,
                enabled=False,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )

    return ExperimentConfig(
        model=model,
        dataset=dataset,
        data_loader=data_loader,
        train_module=train_module,
        trainer=trainer,
    )


def build_pretrain_config(opts: argparse.Namespace) -> ExperimentConfig:
    """Build the OLMo 3 1B stage-1 pretraining configuration."""
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH
    tokenizer = TokenizerConfig.dolma2()

    model = TransformerConfig.olmo3_1B(
        vocab_size=tokenizer.padded_vocab_size(),  # pad to a multiple of 128
        attn_backend=AttentionBackendName.flash_3,
    )

    dataset = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0625_150Bsample,
        tokenizer=tokenizer,
        mix_base_dir=opts.data_root,
        sequence_length=sequence_length,
        max_target_sequence_length=max(8192, sequence_length),
        work_dir=opts.work_dir,
    )

    data_loader = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=SEED,
        num_workers=8,
        prefetch_factor=2,
    )

    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=4 * DEFAULT_SEQUENCE_LENGTH,
        max_sequence_length=sequence_length,
        optim=build_optim_config(
            opts.optim,
            muon_lr=MUON_LR,
            adam_lr=ADAM_LR,
        ),
        scheduler=CosWithWarmup(warmup_steps=2000),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        float8_config=Float8Config(enabled=False),
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
    config.trainer = config.trainer.with_callback(
        "lm_evaluator",
        LMEvaluatorCallbackConfig(
            eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                DataMix.v3_small_ppl_validation,
                mix_base_dir=opts.data_root,
                sequence_length=sequence_length,
                tokenizer=tokenizer,
                work_dir=opts.work_dir,
            ),
            eval_interval=EVAL_LM_STEPS,
            # eval_interval=50,
        ),
    ).with_callback(
        "downstream_evaluator",
        DownstreamEvaluatorCallbackConfig(
            tasks=sorted(FAST_TASKS),
            tokenizer=tokenizer,
            eval_interval=EVAL_DOWN_STEPS,
            # eval_interval=50,
        ),
    )
    config.init_seed = SEED
    return config
