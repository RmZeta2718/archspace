"""Shared base configuration for the OLMo 3 1B training stages."""

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
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
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

# Shared local data-loader policy for all five stages.
seed = 34_521
num_workers = 8
prefetch_factor = 4

EVAL_LM_STEPS = 500
EVAL_DOWN_STEPS = 12500


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


def build_short_context_model(tokenizer: TokenizerConfig) -> TransformerConfig:
    """Build the complete short-context model used by pretraining and midtraining."""
    return TransformerConfig.olmo3_1B(
        vocab_size=tokenizer.padded_vocab_size(),  # pad to a multiple of 128
        attn_backend=AttentionBackendName.flash_3,
    )


def build_optim_config(
    name: str,
    *,
    lr: float,
) -> OptimConfig:
    """Build the pipeline-wide optimizer with the stage learning rate."""
    # All five stages use this local profile so adjacent checkpoints have compatible optimizer
    # types and parameter groups. Published OLMo 3 pre/mid/long runs use SkipStep AdamW with
    # weight decay 0.1, while published SFT uses weight decay 0. This pipeline instead uses 0.033
    # for both Muon and AdamW; the AdamW embedding group remains exempt from weight decay.
    # Both branches inherit optim.compile=False: Muon does not support optimizer-step compilation,
    # and the published OLMo 3 AdamW recipes also keep it disabled. Model compilation is separate.
    # Equivalent whole-object CLI override; define `lr` in the shell first:
    # "--train_module.optim={type: muon, lr: ${lr}, weight_decay: 0.033, betas: [0.9, 0.95]}"
    if name == "muon":
        return MuonConfig(
            lr=lr,
            weight_decay=0.033,
            betas=(0.9, 0.95),
        )
    # Equivalent whole-object CLI override; define `lr` in the shell first:
    # "--train_module.optim={type: skip_step_adamw, lr: ${lr}, weight_decay: 0.033, betas: [0.9, 0.95], group_overrides: [{params: [embeddings.weight], opts: {weight_decay: 0.0}}]}"
    if name == "adam":
        return SkipStepAdamWConfig(
            lr=lr,
            weight_decay=0.033,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight"], opts={"weight_decay": 0.0}
                )
            ],
        )
    raise ValueError(f"Unknown optimizer '{name}'")


def get_ephemeral_save_interval(global_batch_size: int) -> int:
    """Derive a ten-step-aligned checkpoint interval of approximately one Gi tokens."""
    return round(2**30 / global_batch_size / 10) * 10


def configure_stage_continuation(trainer: TrainerConfig) -> None:
    """Configure checkpoint loading for a stage that continues from its parent."""
    # script_utils.main first probes save_folder, where these flags require a full same-stage
    # trainer+optimizer resume. If none exists, ExperimentConfig.load_path loads the parent while
    # explicitly skipping trainer state; optimizer state is retained through load_optim_state=True.
    trainer.load_strategy = LoadStrategy.always
    trainer.load_trainer_state = True
    trainer.load_optim_state = True


def build_common_config(
    opts: argparse.Namespace,
    *,
    model: TransformerConfig,
    dataset: NumpyDatasetConfig,
    data_loader: NumpyDataLoaderConfig,
    train_module: TransformerTrainModuleConfig,
) -> ExperimentConfig:
    """Build an experiment from required stage components and the shared trainer."""
    ephemeral_save_interval = get_ephemeral_save_interval(data_loader.global_batch_size)

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
                save_interval=None,  # No periodic permanent checkpoints; final save remains.
                ephemeral_save_interval=ephemeral_save_interval,
                max_checkpoints=1,
                # Optional local override: skip the initial pre-training checkpoint.
                # CLI: `--trainer.callbacks.checkpointer.pre_train_checkpoint=false`.
                # pre_train_checkpoint=False,
                # Optional local override: force synchronous saves; None auto-selects by backend.
                # CLI: `--trainer.callbacks.checkpointer.save_async=false`.
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
        init_seed=seed,
    )


def build_pretrain_config(
    opts: argparse.Namespace,
    *,
    lr: float,
    sequence_length: int,
    rank_microbatch_size: int,
    global_batch_size: int,
) -> ExperimentConfig:
    """Build the short-context baseline inherited by midtraining."""
    sequence_length = opts.sequence_length or sequence_length
    tokenizer = TokenizerConfig.dolma2()

    model = build_short_context_model(tokenizer)

    # Plain FSL concatenates token arrays into fixed contiguous windows; documents may be split,
    # and it consumes neither packed-document boundaries nor assistant-label-mask sidecars.
    dataset = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0625_150Bsample,
        tokenizer=tokenizer,
        mix_base_dir=opts.data_root,
        sequence_length=sequence_length,
        work_dir=opts.work_dir,
    )

    data_loader = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size,
        seed=seed,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=sequence_length,
        optim=build_optim_config(
            opts.optim,
            lr=lr,
        ),
        scheduler=CosWithWarmup(warmup_steps=2000),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
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
    config.trainer = config.trainer.with_callback(
        "lm_evaluator",
        LMEvaluatorCallbackConfig(
            eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                DataMix.v3_small_ppl_validation,
                tokenizer=tokenizer,
                mix_base_dir=opts.data_root,
                sequence_length=sequence_length,
                work_dir=opts.work_dir,
            ),
            eval_interval=EVAL_LM_STEPS,
        ),
    ).with_callback(
        "downstream_evaluator",
        DownstreamEvaluatorCallbackConfig(
            tasks=sorted(FAST_TASKS),
            tokenizer=tokenizer,
            eval_interval=EVAL_DOWN_STEPS,
        ),
    )
    return config
