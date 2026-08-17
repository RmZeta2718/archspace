# OLMo 3 1B Five-Stage Reproduction

English | [中文](README_zh.md)

This directory provides the training recipes and launcher for the five-stage OLMo 3 1B pipeline:

1. stage 1: pretraining;
2. stage 2: midtraining;
3. stage 3: long-context extension;
4. stage 4: Think SFT;
5. stage 5: Instruct SFT.

OLMo-core provides the model definitions, distributed training, checkpoint I/O, and data loading. The Muon
fixes required by these recipes are pinned in the `third_party/OLMo-core` submodule, and the model
configuration uses `TransformerConfig.olmo3_1B()`.

## 1. Directory structure

```text
reproduce/train-olmo-core/
├── README.md
├── README_zh.md
├── cfgs/
│   ├── _olmo3_1b_base.py
│   ├── _olmo3_1b_long.py
│   ├── OLMo3-1B-pretrain.py
│   ├── OLMo3-1B-midtraining.py
│   ├── OLMo3-1B-long-context.py
│   └── OLMo3-1B-sft.py
├── run/
│   ├── envs.sh.example
│   └── run.sh
└── third_party/
    └── OLMo-core/ # Editable Git submodule
```

## 2. Recipe overview

| Stage   | Data                                  | Sequence length | Rank microbatch (tokens) | Global batch (tokens) | Epochs | Default Muon LR |
| ------- | ------------------------------------- | --------------: | -----------------------: | --------------------: | -----: | --------------: |
| stage 1 | `OLMo_mix_0625_150Bsample`            |           4,096 |                   16,384 |             2,097,152 |      1 |          `5e-3` |
| stage 2 | `OLMo_midtraining_mix_0625_100B`      |           4,096 |                   16,384 |             1,048,576 |      1 | `2.071235285e-4` |
| stage 3 | `OLMo_longmino_mix_0625`              |          32,768 |                   32,768 |             2,097,152 |      1 | `2.071235285e-4` |
| stage 4 | `Dolci-Think-SFT-7B` paired NPY files |          32,768 |                   32,768 |             1,048,576 |      2 |          `5e-5` |
| stage 5 | `Dolci-Instruct-SFT` paired NPY files |          32,768 |                   32,768 |             1,048,576 |      2 |          `8e-5` |

All five stages use BF16, FlashAttention-3, and Muon by default. Stages 1 and 2 use fixed-length datasets.
Stage 3 uses Longmino document packing, an intra-document attention mask, and 8x YaRN RoPE scaling from an
old context length of 4,096. Stages 4 and 5 inherit that long-context model and use packed, assistant-masked
SFT data. All five stages use HSDP with context parallelism disabled.
The default FlashAttention-3 configuration targets Hopper GPUs. Other supported GPUs should switch to
FlashAttention-2 as described in section 3.3.

To select the SkipStep AdamW recipe, add `"--optim=adam"` to `all_stage_args` in `run/run.sh`. All five stages
and every resume attempt in a pipeline must use the same optimizer because later stages inherit the optimizer
state from the preceding stage.

These experimental configurations adapt the official OLMo 3 7B recipes to the 1B model; they have not been
officially released or tuned as OLMo 3 1B recipes.

## 3. Environment setup with uv

This workflow targets Python 3.12, PyTorch 2.10.0, and CUDA 12.8. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) before following the steps below; uv creates
the virtual environments and installs their packages.

For a new ArchSpace checkout, initialize OLMo-core as part of the clone:

```bash
git clone --recurse-submodules --branch arch/base \
  https://github.com/InternLM/archspace.git
cd archspace/reproduce/train-olmo-core
```

For an existing checkout or a clone created without `--recurse-submodules`, initialize OLMo-core from the
workflow directory:

```bash
git -C ../.. submodule sync --recursive
git -C ../.. submodule update --init --recursive
git -C ../.. submodule status --recursive
```

The parent repository pins `third_party/OLMo-core` to commit
`45f248d361f0e292c39298d6fba4b4450aed3cc5`. Run the recursive update command again after the parent
repository updates this gitlink.

### 3.1 Create and activate the training environment

From `reproduce/train-olmo-core`, create the local training environment. If Python 3.12 is unavailable,
uv obtains a compatible interpreter:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
python --version
```

Keep this environment active when running `run/run.sh`; the launcher uses its `python` and `torchrun`.
Reactivate it with `source .venv/bin/activate` in each new shell.

### 3.2 Install PyTorch and OLMo-core

Install the CUDA build of PyTorch and the kernel build tools, then install the pinned OLMo-core submodule in
editable mode:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0 torchvision torchaudio

uv pip install 'setuptools<70' wheel packaging ninja
uv pip install --editable 'third_party/OLMo-core[all]'
```

The editable checkout at `third_party/OLMo-core` is the OLMo-core package used by this workflow, so local
source changes are immediately available to the training commands.

### 3.3 Install attention kernels

Install the pinned FlashAttention-2 version used by the OLMo-core environment:

```bash
MAX_JOBS=8 \
uv pip install --no-build-isolation 'flash-attn==2.8.2'
```

The default five-stage configurations use FlashAttention-3 and target Hopper GPUs such as H100 and H800.
Install FA3 from the source commit used by the OLMo-core environment. `FLASH_ATTENTION_FORCE_BUILD=TRUE`
builds the extension from that source, while `--no-cache` prevents reuse of a wheel built with different
feature flags:

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTENTION_DISABLE_FP16=TRUE \
FLASH_ATTENTION_DISABLE_SM80=TRUE \
MAX_JOBS=8 \
uv pip install --no-cache --no-build-isolation \
  'flash-attn-3 @ git+https://github.com/Dao-AILab/flash-attention.git@92ca9da8d66f7b34ff50dc080ec0fef9661260d6#subdirectory=hopper'
```

For other supported GPUs, use the FA2 installation above and add the following override to `all_stage_args`
in `run/run.sh`:

```bash
"--model.block.sequence_mixer.backend=flash_2"
```

The default HSDP recipe has context parallelism disabled. A custom ring-CP configuration also needs the
version of `ring-flash-attn` used by the OLMo-core environment:

```bash
uv pip install --no-build-isolation 'ring-flash-attn==0.1.8'
```

Adjust `MAX_JOBS` to the CPU and memory available on the build node.

### 3.4 Verify the training environment

Run the checks that correspond to the backends installed on this machine:

```bash
uv pip check
# FA2 (all installations)
python -c 'from olmo_core.nn.attention.flash_attn_api import has_flash_attn_2; assert has_flash_attn_2()'
# FA3 (Hopper only)
python -c 'from olmo_core.nn.attention.flash_attn_api import has_flash_attn_3; assert has_flash_attn_3()'
# ring-flash-attn (optional installation only)
python -c 'from olmo_core.nn.attention.flash_attn_api import has_ring_flash_attn; assert has_ring_flash_attn()'
python -c 'import dion, torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

## 4. Prepare the dataset

Follow the Dataset Card for
[ArchSpace-Collection/OLMo3-1B-Dataset](https://huggingface.co/datasets/ArchSpace-Collection/OLMo3-1B-Dataset)
to select, download, and extract the data. It also documents storage requirements, layout, provenance, and
license.

Set `olmo3_data_root` in `run/envs.sh` to the output root passed to the dataset's `extract.sh`. Place this
directory on shared storage at the same path on every training node.

The selected `run.sh` stages determine which extracted groups must be present:

| `run.sh` stage | Configuration | Dataset Card groups | Evaluator input | Parent checkpoint when starting here |
| -------------- | ------------- | ------------------- | --------------- | ------------------------------------ |
| stage 1 | pretraining | stage 1 and eval | Dolma 2 `tokenizer_json` | — |
| stage 2 | midtraining | stage 2 and eval | Dolma 2 `tokenizer_json` | stage 1 |
| stage 3 | long-context | stage 3 | — | stage 2 |
| stage 4 | Think SFT | stage 4 | — | stage 3 |
| stage 5 | Instruct SFT | stage 5 | — | stage 4 |

The default launcher selects all five stages and therefore uses every dataset group. A narrowed stage loop
uses the corresponding groups. If it starts at stage 2–5, make the preceding stage checkpoint available at
the default pipeline path or through `previous_save_folder`.

## 5. Run training

### 5.1 Configure the environment and recipe

```bash
cd reproduce/train-olmo-core
cp run/envs.sh.example run/envs.sh
# edit run/envs.sh
```

`run/envs.sh` is excluded by `.gitignore`. Configure these machine-specific values before launching:

| Setting | Meaning |
| ------- | ------- |
| `olmo3_data_root` | The dataset extraction output root described in Section 4; use the same shared path on every node. |
| `out_root` | A shared, persistent root for checkpoints, trainer state, W&B files, and dataset caches. |
| `tokenizer_json` | A Dolma 2 `tokenizer.json` readable on every node. Required at launcher startup and used by the stage 1/2 downstream evaluator. |
| `wandb_entity`, `wandb_project`, `WANDB_MODE` | W&B destination and mode; see Section 5.3. |

The launcher accepts two optional positional arguments:

```bash
bash run/run.sh [TIMESTAMP] [BASE_PORT]
```

Configure stage selection, optimizer choice, and Python dotlist overrides in the editable lowercase block
near the top of `run/run.sh`:

| `run.sh` setting | Purpose |
| ---------------- | ------- |
| `pipeline_name` | Names the training output under `${out_root}/runs/`; edit it when starting a distinct recipe. |
| `all_stage_args` | Python CLI overrides applied to every selected stage, such as optimizer or attention backend. |
| `stage1_args` ... `stage5_args` | Overrides owned by one stage, such as that stage's learning rate. |
| `for stage_index in 1 2 3 4 5` | The selected stages and their execution order. |

Keep a given dotlist option in one array. `all_stage_args` is appended after the per-stage arguments, so a
duplicate there can override the stage-specific value. Optimizer changes should normally be made in
`all_stage_args` and kept consistent across the checkpoint chain.

To run only stages 3–5, edit the loop to:

```bash
for stage_index in 3 4 5; do
```

If the first selected stage is stage 2–5, it needs the preceding stage's checkpoint. By default, stage N
loads from `${out_root}/runs/${pipeline_name}/stage(N-1)/checkpoints`. To start from a checkpoint outside
that pipeline, set `previous_save_folder=/path/to/parent/checkpoints` in the corresponding `case` branch in
`run.sh`.

If you change `data_loader.global_batch_size`, edit it in the stage's Python configuration and then dry-run
the whole pipeline. The configuration derives its temporary checkpoint interval from this value before
merging dotlist arguments.

### 5.2 Validate the configuration

Before allocating a full training run, resolve all selected configurations and overrides:

```bash
ENABLE_WANDB=0 DRY_RUN=1 bash run/run.sh 0816_120000 29500
```

Dry-run visits every selected stage, including stages with an existing `_SUCCESS` marker, and leaves training
outputs unchanged. It validates configuration construction. Before training, separately verify the data
shards, parent checkpoints, distributed topology, and ports.

### 5.3 W&B

`envs.sh.example` sets `WANDB_MODE=offline` by default. This mode requires no API key, writes files under each
stage's `trainer/wandb/` directory, and automatically disables remote cancel tags, which only work online.

Disable W&B for a launch with `ENABLE_WANDB=0`. For online logging, set `WANDB_MODE=online`, `wandb_entity`,
and `wandb_project` in `run/envs.sh`, then export `WANDB_API_KEY` in the calling shell.

The timestamp identifies the W&B run segment, while `out_root` and `pipeline_name` identify the training
output. Keep `pipeline_name` unchanged when resuming an experiment, and use a new timestamp for each job
attempt so its W&B segment remains distinct. Every node in one multi-node attempt must use the same
timestamp.

### 5.4 Outputs and shared caches

```text
out_root/
├── dataset-cache/
│   ├── olmo3-stage1/...
│   ├── olmo3-stage2/...
│   ├── olmo3-stage3/...
│   ├── olmo3-stage4/...
│   └── olmo3-stage5/...
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
    ├── stage3/
    │   └── ...
    ├── stage4/
    │   └── ...
    └── stage5/
        └── ...
```

`config.json` is the effective configuration, while `data_paths.txt` records the expanded data files that
were actually used. Preserve both together with the W&B records when archiving a reproduction run. The
configurations write a temporary checkpoint approximately every 1 billion tokens, retain only one temporary
checkpoint, and save a final checkpoint at the end of each stage.

Node rank 0 creates `_SUCCESS` after `torchrun` exits successfully for that stage. Treat it as a process
completion marker; inspect the checkpoint and W&B records for the saved step and metrics.

`pipeline_name` namespaces stage outputs under `${out_root}/runs/`. Dataset caches use the separate path
`${out_root}/dataset-cache/olmo3-stageN`. When source data, packing, or configuration fingerprints change,
use a new `out_root` or assign a new cache namespace through `data_work_dir` in `run.sh`.

### 5.5 Resume and stage transitions

> **Compatibility note:** This five-stage recipe uses dataset fingerprints and data-loader state that differ
> from the earlier three-stage recipe. Start it with a new `pipeline_name` and cache namespace. Reuse earlier
> stage outputs, caches, or `_SUCCESS` markers only after validating the model, optimizer, data, and loader
> state together.

To resume within the five-stage recipe, keep `out_root` and `pipeline_name` unchanged and choose a new
attempt timestamp:

```bash
bash run/run.sh 0809_093000
```

The launcher and OLMo-core behave as follows:

1. A `stageN/_SUCCESS` marker skips that stage during training.
2. Otherwise, a checkpoint under the current stage's `checkpoints/` restores the model, optimizer, trainer,
   data-loader, and RNG states.
3. For stages 2–5 with neither marker nor same-stage checkpoint, the top-level `--load_path` initializes the
   model and optimizer from the preceding stage without inheriting its step or epoch progress.
4. Stage 1 starts from scratch when it has neither marker nor checkpoint.

Therefore, rerunning after a stage 2 interruption skips the completed stage 1 and continues from stage 2's
own latest checkpoint. Each later stage starts only after its parent finishes, forming the chain
stage 1 → stage 2 → stage 3 → stage 4 → stage 5.

## 6. Evaluate with OLMES

Training produces OLMo-core distributed checkpoints, while OLMES expects a Hugging Face model directory for
a local model. Convert the checkpoint first, then run OLMES. For base-model evaluation, normally use the
final stage 3 checkpoint. For chat or instruction-following evaluation, convert the relevant stage 4 or
stage 5 checkpoint and select a suite appropriate to that model stage. Convert stages separately when making
stage-to-stage comparisons.

### 6.1 Convert to Hugging Face format

Set `CHECKPOINT` to a specific `step<N>` directory:

```bash
export CHECKPOINT=/path/to/out_root/runs/olmo3-1b/stage3/checkpoints/stepNNNNN
export HF_MODEL_DIR=/path/to/out_root/hf/olmo3-1b-stage3-stepNNNNN

python third_party/OLMo-core/src/examples/huggingface/convert_checkpoint_to_hf.py \
  --checkpoint-input-path "${CHECKPOINT}" \
  --huggingface-output-dir "${HF_MODEL_DIR}" \
  --max-sequence-length 32768
```

The converter reconstructs the OLMo 3 architecture from the checkpoint's `config.json` and uses
`allenai/dolma2-tokenizer` from the configuration by default. In an offline environment, pass
`--tokenizer /path/to/local/hf-tokenizer-directory` with a complete directory loadable by
`AutoTokenizer.from_pretrained()`.

Keep the default numerical validation enabled. After conversion, run a minimal loading test:

```bash
python -c 'import os; from transformers import AutoModelForCausalLM, AutoTokenizer; p=os.environ["HF_MODEL_DIR"]; AutoTokenizer.from_pretrained(p); AutoModelForCausalLM.from_pretrained(p); print("HF checkpoint OK")'
```

### 6.2 Install and run OLMES

OLMES requires PyTorch 2.8 and therefore uses a separate uv environment. From
`reproduce/train-olmo-core`, install OLMES with its optional vLLM dependencies from the selected source
commit:

```bash
uv venv --python 3.12 venv/olmes
uv pip install --python venv/olmes/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 'torch==2.8.0'
uv pip install --python venv/olmes/bin/python \
  'ai2-olmes[gpu] @ git+https://github.com/allenai/olmes.git@5a51f502d463b8cdc4a2dcad7d7096c41ff1197e'
uv pip check --python venv/olmes/bin/python
source venv/olmes/bin/activate
```

The Git URL pins the OLMES source itself. Record
`uv pip freeze --python venv/olmes/bin/python` with the evaluation results to capture its resolved
transitive dependencies.

For a small-scale stage 3 base-model experiment, start with the OLMo 3 base-easy suites:

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
conversion arguments, and output directory. Use the configuration's `FAST_TASKS` and PPL in-loop evaluations
for training monitoring, and use a version-pinned OLMES run for final evaluation. For stage 4/5 checkpoints,
choose and record a suitable chat or instruction-following suite.
