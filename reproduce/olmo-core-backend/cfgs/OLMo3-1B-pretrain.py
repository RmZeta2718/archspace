"""
OLMo 3 1B stage-1 pretraining configuration for the local 150B data sample.

This is a 1B adaptation of the official OLMo-3-1025-7B stage-1 recipe in
``src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-1.py``. OLMo 3 does not
publish an official tuned 1B pretraining recipe, so the batch size, learning
rate, and warmup below intentionally retain the official 7B values.
"""

import argparse
from typing import List

from _olmo3_1b import build_pretrain_config, get_olmo3_1b_cli_parser
from olmo_core.script_utils import ExperimentConfig, main


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    """Build the OLMo 3 1B stage-1 pretraining configuration."""
    # This complete stage-1 recipe, including the padded vocabulary size, is also
    # the baseline imported by stage 2.
    # Merge CLI overrides only after the shared defaults have been assembled.
    return build_pretrain_config(opts).merge(overrides)


if __name__ == "__main__":
    main(build_config, parser=get_olmo3_1b_cli_parser())
