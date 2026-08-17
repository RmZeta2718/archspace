"""
OLMo 3 1B stage-2 midtraining configuration.

This is a 1B adaptation of the official OLMo-3-1025-7B midtraining recipe in
`src/scripts/official/OLMo3/OLMo-3-1025-7B-midtrain.py`. OLMo 3 does not publish an officially
tuned 1B recipe. The local pipeline adds Muon and uses weight decay 0.033 instead of the published
AdamW value 0.1; its Python-default LR follows the published value of approximately 2.071e-4.
"""

import argparse
from typing import List

from _olmo3_1b_base import (
    build_pretrain_config,
    configure_stage_continuation,
    get_olmo3_1b_cli_parser,
)

from olmo_core.data import DataMix
from olmo_core.optim import LinearWithWarmup
from olmo_core.script_utils import ExperimentConfig, main
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,  # noqa: F401 - optional commented config below
    TransformerActivationCheckpointingMode,  # noqa: F401 - optional commented config below
)

# Default variables for this stage; supported overrides are shown inline.
lr = 0.00020712352850360292  # Override with `--train_module.optim.lr=1e-3`.
sequence_length = 4096  # Override with `--sequence-length=8192`.
# Override with `--train_module.rank_microbatch_size=8192`.
rank_microbatch_size = 4 * sequence_length
# DO NOT override with `--data_loader.global_batch_size=...`; the ephemeral checkpoint interval
# depends on this value. Edit this default directly instead.
global_batch_size = 2**20  # 1M tokens


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    """Apply the midtraining delta and CLI overrides to the short-context baseline."""
    config = build_pretrain_config(
        opts,
        lr=lr,
        sequence_length=sequence_length,
        rank_microbatch_size=rank_microbatch_size,
        global_batch_size=global_batch_size,
    )

    # Model shape, batching, callbacks, and optimizer parameter groups remain identical to stage 1.
    # The data mix, scheduler, and checkpoint continuation policy are stage-specific.
    config.dataset.mix = DataMix.OLMo_midtraining_mix_0625_100B

    # Optimizer state is restored from stage 1, so the selected recipe must match.
    config.train_module.scheduler = LinearWithWarmup(warmup=0, alpha_f=0.0)
    # Official 7B activation checkpointing; intentionally disabled in this local 1B pipeline.
    # CLI: `'--train_module.ac_config={mode: selected_modules, modules: ["blocks.*.feed_forward"]}'`.
    # config.train_module.ac_config = TransformerActivationCheckpointingConfig(
    #     mode=TransformerActivationCheckpointingMode.selected_modules,
    #     modules=["blocks.*.feed_forward"],
    # )

    configure_stage_continuation(config.trainer)
    return config.merge(overrides)


if __name__ == "__main__":
    main(build_config, parser=get_olmo3_1b_cli_parser())
