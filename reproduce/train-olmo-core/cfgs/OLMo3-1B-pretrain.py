"""
OLMo 3 1B stage-1 pretraining configuration for the local 150B data sample.

This is a 1B adaptation of the official OLMo-3-1025-7B stage-1 recipe in
``src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-1.py``. OLMo 3 does not publish an
officially tuned 1B recipe. The local pipeline adds Muon and uses weight decay 0.033 instead of
the published AdamW value 0.1; its Python-default LR is 5e-3 instead of the published 3e-4.
"""

import argparse
from typing import List

from _olmo3_1b_base import build_pretrain_config, get_olmo3_1b_cli_parser

from olmo_core.script_utils import ExperimentConfig, main

# Default variables for this stage; supported overrides are shown inline.
lr = 5e-3  # Override with `--train_module.optim.lr=3e-3`.
sequence_length = 4096  # Override with `--sequence-length=8192`.
# Override with `--train_module.rank_microbatch_size=8192`.
rank_microbatch_size = 4 * sequence_length
# DO NOT override with `--data_loader.global_batch_size=...`; the ephemeral checkpoint interval
# depends on this value. Edit this default directly instead.
global_batch_size = 2**21  # 2M tokens


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    """Apply CLI overrides to the short-context defaults."""
    # This complete stage-1 recipe, including the padded vocabulary size, is also
    # the baseline imported by stage 2.
    # Merge CLI overrides only after the shared defaults have been assembled.
    config = build_pretrain_config(
        opts,
        lr=lr,
        sequence_length=sequence_length,
        rank_microbatch_size=rank_microbatch_size,
        global_batch_size=global_batch_size,
    )
    return config.merge(overrides)


if __name__ == "__main__":
    main(build_config, parser=get_olmo3_1b_cli_parser())
