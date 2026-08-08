"""
OLMo 3 1B stage-2 midtraining configuration.

This is a 1B adaptation of the official OLMo-3-1025-7B midtraining recipe in
`src/scripts/official/OLMo3/OLMo-3-1025-7B-midtrain.py`. OLMo 3 does not
publish an officially tuned 1B midtraining recipe, so the data schedule and
optimization settings below intentionally retain the official 7B values.
"""

import argparse
from typing import List

from _olmo3_1b import build_optim_config, build_pretrain_config, get_olmo3_1b_cli_parser

from olmo_core.data import DataMix
from olmo_core.optim import LinearWithWarmup
from olmo_core.script_utils import ExperimentConfig, main
from olmo_core.train.common import LoadStrategy

# MAX_TOKENS = 100_000_000_000  # 100B
# Muon retains the 1B recipe; AdamW follows the official stage-2 schedule.
MUON_LR = 5e-4
ADAM_LR = 5e-4
SEED = 1337


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    """Build stage 2 by applying its differences to the pretraining configuration."""
    config = build_pretrain_config(opts)

    # Model shape, including the padded vocabulary size, batching, and callbacks
    # remain identical to stage 1. Only data order and optimization are stage-specific.
    config.dataset.mix = DataMix.OLMo_midtraining_mix_0625_100B
    config.data_loader.seed = SEED

    # Optimizer state is restored from stage 1, so the selected recipe must match.
    config.train_module.optim = build_optim_config(
        opts.optim,
        muon_lr=MUON_LR,
        adam_lr=ADAM_LR,
    )
    config.train_module.scheduler = LinearWithWarmup(warmup=0, alpha_f=0.0)

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
