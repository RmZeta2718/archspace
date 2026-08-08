# OLMo 3 1B Three-Stage Reproduction

English | [中文](README_zh.md)

This directory provides the training recipes and launcher for the three-stage OLMo 3 1B pipeline:

1. stage 1: pretraining;
2. stage 2: midtraining;
3. stage 3: long-context extension.

The model implementation, distributed trainer, checkpoint I/O, and dataset implementation all come from
OLMo-core. This directory contains only the configuration and entry point required for a specific
reproduction, keeping the experiment recipe separate from the general-purpose training framework. Changes
to recipes in this directory do not affect OLMo-core, while framework upgrades and fixes do not require
copying the entire framework into this repository.

This reproduction must use the following custom OLMo-core source branch instead of the general PyPI release:

- [https://github.com/JT-Ushio/OLMo-core-muon-fix/tree/ready_for_archspace_base](https://github.com/JT-Ushio/OLMo-core-muon-fix/tree/ready_for_archspace_base)

This branch contains the Muon fixes required by the recipes. The OLMo 3 model is already implemented by
`TransformerConfig.olmo3_1B()`, so no additional modeling files are needed here.

## 1. Directory structure

```text
reproduce/olmo-core-backend/
├── README.md
├── README_zh.md
├── requirements.txt
├── cfgs/
│   ├── _olmo3_1b.py
│   ├── OLMo3-1B-pretrain.py
│   ├── OLMo3-1B-midtraining.py
│   └── OLMo3-1B-long-context.py
└── run/
    ├── envs.sh.example
    └── run.sh
```

## 2. Recipe overview


| Stage   | Data mix                         | Sequence length | Global batch (tokens) | Parallelism | Default Muon LR |
| --------- | ---------------------------------- | ----------------: | ----------------------: | ------------- | ----------------: |
| stage 1 | `OLMo_mix_0625_150Bsample`       |           4,096 |             2,097,152 | HSDP        |          `1e-3` |
| stage 2 | `OLMo_midtraining_mix_0625_100B` |           4,096 |             2,097,152 | HSDP        |          `5e-4` |
| stage 3 | `OLMo_longmino_mix_0625`         |          65,536 |             4,194,304 | HSDP        |          `5e-4` |

All three stages use BF16, FlashAttention-3, and Muon by default, and each trains for one complete data
epoch. Stages 1 and 2 use fixed-length datasets. Stage 3 uses document packing, an intra-document attention
mask, and 8x YaRN RoPE scaling. All three stages use HSDP without context parallelism.
The default FlashAttention-3 configuration targets Hopper GPUs. Other supported GPUs should switch to
FlashAttention-2 as described in section 3.2.

You can also select the SkipStep AdamW recipe with `adam`, but all three stages and every resume attempt in a
pipeline must use the same optimizer because later stages inherit the optimizer state from the preceding
stage.

These are experimental configurations scaled from the official OLMo 3 7B recipes to the 1B model. They are
not officially released or tuned OLMo 3 1B recipes.

## 3. Environment setup

This project uses PyTorch 2.10 and CUDA 12.8, although any versions compatible with OLMo-core and the other
dependencies should work in principle.

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0 torchvision torchaudio
```

### 3.1 Install the custom OLMo-core from source

We recommend keeping a separate source checkout and installing it in editable mode:

```bash
export OLMO_CORE_SRC=/path/to/OLMo-core-muon-fix
git clone --branch ready_for_archspace_base --single-branch \
  https://github.com/JT-Ushio/OLMo-core-muon-fix.git "${OLMO_CORE_SRC}"

pip install -e "${OLMO_CORE_SRC}[all]"
```

### 3.2 Install attention kernels

Install the latest attention kernels without pinning a tag, commit, or package version. First install
FlashAttention-2:

```bash
MAX_JOBS=8 python -m pip install --upgrade --no-build-isolation flash-attn
```

Hopper GPUs such as H100 and H800 can use FlashAttention-3. Install it from the `hopper/` directory on the
default FlashAttention branch:

```bash
export FLASH_ATTN_SRC=/path/to/flash-attention
git clone --depth 1 --recurse-submodules --shallow-submodules \
  https://github.com/Dao-AILab/flash-attention.git "${FLASH_ATTN_SRC}"

cd "${FLASH_ATTN_SRC}/hopper"
FLASH_ATTENTION_DISABLE_FP16=TRUE \
FLASH_ATTENTION_DISABLE_SM80=TRUE \
MAX_JOBS=8 \
python setup.py install
cd -
```

Other supported GPUs should use FlashAttention-2. All three upstream-synchronized cfgs select `flash_3` by
default. When using FA2, add the following override to the `extra_args` array in `run/run.sh` so it applies to
all three stages:

```bash
"--model.attn_backend=flash_2"
```

`ring-flash-attn` remains available as an optional backend. Install it when a custom configuration enables
ring context parallelism; the current three-stage HSDP recipe without CP does not require it:

```bash
pip install ring-flash-attn
```

Adjust `MAX_JOBS` to the CPU and memory available on the build node. Run the checks that correspond to the
backends you installed:

```bash
python -m pip check
# FA2 (all installations)
python -c 'from olmo_core.nn.attention.flash_attn_api import has_flash_attn_2; assert has_flash_attn_2()'
# FA3 (Hopper only)
python -c 'from olmo_core.nn.attention.flash_attn_api import has_flash_attn_3; assert has_flash_attn_3()'
# ring-flash-attn (optional installation only)
python -c 'from olmo_core.nn.attention.flash_attn_api import has_ring_flash_attn; assert has_ring_flash_attn()'
python -c 'import dion, torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

## 4. Data preparation

### 4.1 Data format

The configurations directly use four `DataMix` manifests installed with the OLMo-core package. Every `.npy`
path listed by these manifests must be a one-dimensional token-ID binary array following the OLMo-core
convention and readable with `numpy.memmap`. Arbitrary text files, or files merely renamed to `.npy`, will not
work. The Dolma 2 tokenizer has a vocabulary size of 100,278, so these recipes infer the array dtype as
`uint32`. Documents must be correctly separated with the EOS token (ID `100257`), because stage 3 document
packing and intra-document masking depend on these boundaries.

The data comes from the official OLMo release. This repository will provide a Hugging Face redistribution:
LINK TODO.

### 4.2 Expected `olmo3_data_root` layout

`run.sh` passes `olmo3_data_root` from `envs.sh` unchanged to all three configurations. OLMo-core then uses it
as the prefix for every relative path in the manifests. The approximate directory layout is shown below;
ellipses represent all sources and shards listed in the manifests:

```text
olmo3_data_root/
├── preprocessed/
│   ├── dolma2-0625/v0.1-150b/
│   │   └── allenai/dolma2-tokenizer/
│   │       ├── finemath-3plus/part-000-00000.npy
│   │       └── ...
│   ├── dolma3-dolmino-official/100B/
│   │   └── allenai/dolma3-tokenizer/
│   │       ├── code-meta-reasoning/part-00-00000.npy
│   │       └── ...
│   └── dolma3_longmino_0625/
│       └── allenai/dolma3-tokenizer/
│           ├── 000000.npy
│           └── ...
└── eval-data/perplexity/
    └── v3_small_dolma2-tokenizer/
        ├── c4_en/val/part-0-00000.npy
        ├── dolma_books/val/part-0-00000.npy
        └── ...
```

Stage 1 uses the first tree, stage 2 the second, and stage 3 the third. The in-loop LM evaluations in stages 1
and 2 also require the final validation tree. Every manifest filename must match exactly; providing only
similar top-level directories is insufficient.

`tokenizer_json` is another required path. It must point to a Dolma 2 `tokenizer.json` readable by every node
and is used by the stage 1 and 2 in-loop downstream evaluator. It does not replace the tokenized training
arrays described above.

## 5. Run training

```bash
cd reproduce/olmo-core-backend
cp run/envs.sh.example run/envs.sh
# edit run/envs.sh
bash run/run.sh
```

`run/envs.sh` is excluded by `.gitignore`.

### 5.1 W&B

`envs.sh.example` sets `WANDB_MODE=offline` by default. This mode requires no API key, writes files under each
stage's `trainer/wandb/` directory, and automatically disables remote cancel tags, which only work online.

The timestamp is used only in the W&B run name and ID. Keep `pipeline_name` unchanged when resuming the same
experiment, but use a new timestamp for each new job attempt to prevent the new W&B segment from overwriting
or mixing with the previous attempt. Every node in the same multi-node attempt must use the same timestamp.

### 5.2 Output structure

```text
out_root/
├── dataset-cache/
│   ├── olmo3-stage1/...
│   ├── olmo3-stage2/...
│   └── olmo3-stage3/...
└── runs/olmo3-1b/
    ├── stage1/
    │   ├── _SUCCESS
    │   ├── checkpoints/
    │   │   └── step<N>/
    │   │       ├── .metadata.json
    │   │       ├── config.json
    │   │       ├── data_paths.txt
    │   │       ├── model_and_optim/
    │   │       │   ├── .metadata
    │   │       │   └── __<rank>_<shard>.distcp
    │   │       └── train/
    │   │           └── rank<rank>.pt
    │   └── trainer/wandb/...
    ├── stage2/
    │   └── ...
    └── stage3/
        └── ...
```

`config.json` is the effective configuration, while `data_paths.txt` records the expanded data files that
were actually used. Preserve both together with the W&B records when archiving a reproduction run. The
configurations write a temporary checkpoint approximately every 1 billion tokens, retain only one temporary
checkpoint, and save a final checkpoint at the end of each stage.

Node rank 0 creates `_SUCCESS` after `torchrun` exits successfully for that stage. It indicates successful
process completion; it does not revalidate the checkpoint step or metric values.

### 5.3 Resume and stage transitions

A normal resume does not require specifying a checkpoint manually:

```bash
# Keep out_root and pipeline_name unchanged; use a new attempt timestamp.
bash run/run.sh 0809_093000
```

The launcher and OLMo-core behave as follows:

1. If `stageN/_SUCCESS` exists, that stage is skipped.
2. If `_SUCCESS` does not exist but the current stage has a checkpoint under `checkpoints/`, the model,
   optimizer, trainer, data-loader, and RNG states are restored from it.
3. If the current stage 2 or 3 has no checkpoint, the top-level `--load_path` initializes the model and
   optimizer from the preceding stage's checkpoint without inheriting that stage's step or epoch progress.
4. If the current stage 1 has no checkpoint, training starts from scratch.

Therefore, rerunning after a stage 2 interruption skips the completed stage 1 and continues from stage 2's
own latest checkpoint. Stage 3 starts only after stage 2 finishes.

## 6. Evaluate with OLMES

Training produces OLMo-core distributed checkpoints, while OLMES expects a Hugging Face model directory for
a local model. Convert the checkpoint first, then run OLMES. Normally, you evaluate the final stage 3
checkpoint. To compare stages, convert stages 1, 2, and 3 separately.

### 6.1 Convert to Hugging Face format

Select a specific `step<N>` directory, not its parent `checkpoints/` directory:

```bash
export CHECKPOINT=/path/to/out_root/runs/olmo3-1b/stage3/checkpoints/step11921
export HF_MODEL_DIR=/path/to/out_root/hf/olmo3-1b-stage3-step11921

python "${OLMO_CORE_SRC}/src/examples/huggingface/convert_checkpoint_to_hf.py" \
  --checkpoint-input-path "${CHECKPOINT}" \
  --huggingface-output-dir "${HF_MODEL_DIR}" \
  --max-sequence-length 65536
```

The converter reconstructs the OLMo 3 architecture from the checkpoint's `config.json` and uses
`allenai/dolma2-tokenizer` from the configuration by default. In an offline environment, additionally pass
`--tokenizer /path/to/local/hf-tokenizer-directory`. This must be a complete directory loadable by
`AutoTokenizer.from_pretrained()`, not an individual `tokenizer.json` file.

Numerical validation is enabled by default. Avoid `--skip-validation` unless you have validated the result
separately and explicitly accept the risk. After conversion, run a minimal loading test:

```bash
python -c 'import os; from transformers import AutoModelForCausalLM, AutoTokenizer; p=os.environ["HF_MODEL_DIR"]; AutoTokenizer.from_pretrained(p); AutoModelForCausalLM.from_pretrained(p); print("HF checkpoint OK")'
```

### 6.2 Install and run OLMES

Use a separate evaluation environment so that the vLLM and Transformers versions do not affect the training
environment:

```bash
git clone https://github.com/allenai/olmes.git /path/to/olmes
cd /path/to/olmes
python -m pip install -e '.[gpu]'
git rev-parse HEAD
```

For a small-scale experiment, start with the OLMo 3 base-easy suites:

```bash
olmes \
  --model "${HF_MODEL_DIR}" \
  --task \
    olmo3:base_easy:code_bpb \
    olmo3:base_easy:math_bpb \
    olmo3:base_easy:qa_rc \
    olmo3:base_easy:qa_bpb \
  --output-dir /path/to/out_root/evals/olmo3-1b-stage3-base-easy
```

If the installed OLMES/vLLM versions support this model, add `--model-type vllm` for higher throughput. A
formal report should preserve the OLMES commit, complete command, task suite, checkpoint step, Hugging Face
conversion arguments, and output directory. The `FAST_TASKS` and PPL in-loop evaluations built into the
training configuration are intended for training monitoring and do not replace a final, version-pinned OLMES
evaluation.
