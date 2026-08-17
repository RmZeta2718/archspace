"""
OLMo 3 1B Think SFT (stage 4) and Instruct SFT (stage 5).

OLMo 3 publishes these post-training hyperparameters for 7B, not 1B. This configuration keeps
the disclosed 7B sequence length, token batch, duration, packing, masking, and schedule while
substituting the local OLMo 3 1B long-context architecture and locally selected phase LRs.

Sources:

* OLMo 3 Appendix A.6.1 / Table 47: https://arxiv.org/html/2512.13961v2#A6.SS1
* Open Instruct Think launcher:
  https://github.com/allenai/open-instruct/blob/5fb2acc161b572201628d50cc7145d109edac140/scripts/train/olmo3/7b_think_sft.sh
* Open Instruct Instruct launcher:
  https://github.com/allenai/open-instruct/blob/5fb2acc161b572201628d50cc7145d109edac140/scripts/train/olmo3/7b_instruct_sft.sh
* Published OLMo-core Think implementation:
  https://github.com/allenai/OLMo-core/blob/38f66526c9d1ba6b97269ebfb429749a5feb528f/src/scripts/train/sft/OLMo2-7B-sft.py
* Published OLMo-core Instruct implementation:
  https://github.com/allenai/OLMo-core/blob/9e97471057d7046f0ae7315e0225d117b54186f9/src/scripts/train/sft/OLMo-sft.py

The optimizer choice is intentionally pipeline-wide. Published SFT uses SkipStep AdamW with
weight decay 0; this pipeline adds Muon and gives both optimizer choices weight decay 0.033, with
an AdamW no-decay embedding group, so optimizer state remains compatible across all five stages.
"""

import argparse
from typing import List

from _olmo3_1b_base import get_olmo3_1b_cli_parser
from _olmo3_1b_long import build_long_context_config, source_group_size

from olmo_core.data import NumpyPackedFSLDatasetConfig, TokenizerConfig
from olmo_core.optim import LinearWithWarmup
from olmo_core.script_utils import ExperimentConfig, main
from olmo_core.train import Duration

# These subdirectories match the local Open Instruct conversion layout. Each contains paired
# token_ids_part_*.npy and labels_mask_part_*.npy files.
THINK_DATASET_SUBDIR = "Dolci-Think-SFT-7B"
INSTRUCT_DATASET_SUBDIR = "Dolci-Instruct-SFT"

# Default variables for these SFT stages; supported overrides are shown inline.
think_lr = 5e-5  # Override Think SFT with `--train_module.optim.lr=1e-4`.
instruct_lr = 8e-5  # Override Instruct SFT with `--train_module.optim.lr=1e-4`.
sequence_length = 32768  # Override with `--sequence-length=16384`.
rank_microbatch_size = (
    sequence_length  # Override with `--train_module.rank_microbatch_size=16384`.
)
# DO NOT override with `--data_loader.global_batch_size=...`; the ephemeral checkpoint interval
# depends on this value. Edit this default directly instead.
global_batch_size = 2**20  # 1M tokens
# With FlashAttention 3 and CP disabled, B / rank_microbatch_size = 32, so DP must divide 32.


def get_sft_cli_parser() -> argparse.ArgumentParser:
    """Build the CLI shared by the Think and Instruct SFT phases."""
    parser = get_olmo3_1b_cli_parser()
    parser.add_argument(
        "--sft-stage",
        choices=("think", "instruct"),
        required=True,
        help="Select the Think or Instruct SFT recipe.",
    )
    return parser


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    """Apply the selected SFT delta and CLI overrides to the long-context baseline."""
    if opts.sft_stage == "think":
        lr = think_lr
        dataset_subdir = THINK_DATASET_SUBDIR
    elif opts.sft_stage == "instruct":
        lr = instruct_lr
        dataset_subdir = INSTRUCT_DATASET_SUBDIR
    else:
        raise ValueError(f"Unknown SFT stage '{opts.sft_stage}'")

    # The model, loader, optimizer, parallelism, clipping, and microbatch settings are inherited
    # from the long-context baseline.
    config = build_long_context_config(
        opts,
        lr=lr,
        sequence_length=sequence_length,
        rank_microbatch_size=rank_microbatch_size,
        global_batch_size=global_batch_size,
    )
    resolved_sequence_length = config.train_module.max_sequence_length

    # Open Instruct has already applied the chat template. Masked SFT packed data pairs token IDs
    # with assistant-only trainable-token masks, uses the default truncate strategy for overlong
    # conversations, and records document boundaries for isolated attention. Eight consecutive
    # token/mask shard pairs in resolved path order share each OBFD packing pool.
    dataset_path = f"{opts.data_root.rstrip('/')}/{dataset_subdir}"
    config.dataset = NumpyPackedFSLDatasetConfig(
        tokenizer=TokenizerConfig.dolma2(),
        paths=[f"{dataset_path}/token_ids_part_*.npy"],
        label_mask_paths=[f"{dataset_path}/labels_mask_part_*.npy"],
        expand_glob=True,
        sequence_length=resolved_sequence_length,
        generate_doc_lengths=True,  # enables intra-document masking
        source_group_size=source_group_size,
        work_dir=opts.work_dir,
    )

    config.train_module.scheduler = LinearWithWarmup(
        warmup_fraction=0.03,
        alpha_f=0.0,
    )
    config.train_module.z_loss_multiplier = None

    # Keep the callback/checkpointer profile common to stages 1-3; only SFT duration differs.
    # Two epochs follow OLMo 3 Appendix A.6.1 / Table 47.
    config.trainer.max_duration = Duration.epochs(2)
    config = config.merge(overrides)
    if config.dataset.sequence_length != config.train_module.max_sequence_length:
        raise ValueError(
            "SFT dataset sequence_length must match train_module.max_sequence_length; "
            "use --sequence-length to change both"
        )
    return config


if __name__ == "__main__":
    main(build_config, parser=get_sft_cli_parser())
