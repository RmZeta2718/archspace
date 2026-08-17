"""
OLMo 3 1B stage-3 long-context extension configuration.

This is a 1B adaptation of the OLMo 3 7B long-context recipe in
`src/scripts/official/OLMo3/OLMo-3-1025-7B-long-context.py`. OLMo 3 does not publish an officially
tuned 1B recipe. The local pipeline adds Muon and uses weight decay 0.033 instead of the published
AdamW value 0.1; its Python-default LR follows the published value of approximately 2.071e-4.
"""

import argparse
from typing import List

from _olmo3_1b_base import get_olmo3_1b_cli_parser
from _olmo3_1b_long import build_long_context_config

from olmo_core.script_utils import ExperimentConfig, main

# Default variables for this stage; supported overrides are shown inline.
lr = 0.00020712352850360292  # Override with `--train_module.optim.lr=1e-3`.
sequence_length = 32768  # Override with `--sequence-length=16384`.
rank_microbatch_size = (
    sequence_length  # Override with `--train_module.rank_microbatch_size=16384`.
)
# DO NOT override with `--data_loader.global_batch_size=...`; the ephemeral checkpoint interval
# depends on this value. Edit this default directly instead.
global_batch_size = 2**21  # 2M tokens


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    """Apply CLI overrides to the long-context defaults."""
    config = build_long_context_config(
        opts,
        lr=lr,
        sequence_length=sequence_length,
        rank_microbatch_size=rank_microbatch_size,
        global_batch_size=global_batch_size,
    )
    return config.merge(overrides)


if __name__ == "__main__":
    main(build_config, parser=get_olmo3_1b_cli_parser())
