# OLMo 3 1B 五阶段复现

[English](README.md) | 中文

本目录提供 OLMo 3 1B 的五阶段训练配方与启动脚本：

1. stage 1：pretraining；
2. stage 2：midtraining；
3. stage 3：long-context extension；
4. stage 4：Think SFT；
5. stage 5：Instruct SFT。

模型定义、分布式训练、checkpoint I/O 和数据加载均由 OLMo-core 提供。本配方所需的 Muon 修复
固定在 `third_party/OLMo-core` Git 子模块中，模型配置使用
`TransformerConfig.olmo3_1B()`。

## 1. 目录结构

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
    └── OLMo-core/ # 以可编辑模式安装的 Git 子模块
```

## 2. 配方概览

| 阶段    | 数据                                  | 序列长度 | rank microbatch（token） | 全局 batch（token） | epoch | 默认 Muon LR |
| ------- | ------------------------------------- | -------: | ----------------------: | --------------------: | ----: | -------------: |
| stage 1 | `OLMo_mix_0625_150Bsample`            |    4,096 |                  16,384 |             2,097,152 |     1 |         `5e-3` |
| stage 2 | `OLMo_midtraining_mix_0625_100B`      |    4,096 |                  16,384 |             1,048,576 |     1 | `2.071235285e-4` |
| stage 3 | `OLMo_longmino_mix_0625`              |   32,768 |                  32,768 |             2,097,152 |     1 | `2.071235285e-4` |
| stage 4 | `Dolci-Think-SFT-7B` 成对 NPY 文件 |   32,768 |                  32,768 |             1,048,576 |     2 |         `5e-5` |
| stage 5 | `Dolci-Instruct-SFT` 成对 NPY 文件 |   32,768 |                  32,768 |             1,048,576 |     2 |         `8e-5` |

五个阶段默认都使用 BF16、FlashAttention-3 和 Muon。stage 1/2 使用固定长度数据集；
stage 3 使用 Longmino 文档打包和文档内 attention mask，并以 4,096 为原始上下文长度进行 8 倍
YaRN RoPE 缩放。stage 4/5 继承该长上下文模型，并使用打包且带 assistant mask 的 SFT 数据。
五个阶段均使用 HSDP，并关闭上下文并行。
默认的 FlashAttention-3 配置面向 Hopper GPU；其他支持的 GPU 应按 3.3 节切换到 FlashAttention-2。

如需选择 SkipStep AdamW 配方，请在 `run/run.sh` 的 `all_stage_args` 中加入 `"--optim=adam"`。
同一流水线的五个阶段及所有续训任务必须使用同一种优化器，因为后续阶段会继承前一阶段的
优化器状态。

这些实验配置由 OLMo 3 7B 官方配方缩放至 1B 模型，尚未作为正式调优的 OLMo 3 1B 配方发布。

## 3. 使用 uv 安装环境

本工作流面向 Python 3.12、PyTorch 2.10.0 和 CUDA 12.8。请先安装
[uv](https://docs.astral.sh/uv/getting-started/installation/)；后续步骤由 uv 创建虚拟环境并安装依赖。

首次克隆 ArchSpace 时，可同时初始化 OLMo-core：

```bash
git clone --recurse-submodules --branch arch/base \
  https://github.com/InternLM/archspace.git
cd archspace/reproduce/train-olmo-core
```

已有仓库或首次 clone 未使用 `--recurse-submodules` 时，在当前工作流目录手动初始化 OLMo-core：

```bash
git -C ../.. submodule sync --recursive
git -C ../.. submodule update --init --recursive
git -C ../.. submodule status --recursive
```

父仓库将 `third_party/OLMo-core` 固定在 commit
`45f248d361f0e292c39298d6fba4b4450aed3cc5`。父仓库更新该 gitlink 后，再执行一次递归更新命令
即可同步。

### 3.1 创建并激活训练虚拟环境

在 `reproduce/train-olmo-core` 目录创建训练环境；如本机缺少 Python 3.12，uv 会自动获取兼容的
解释器：

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
python --version
```

运行 `run/run.sh` 时应保持该环境处于激活状态，启动脚本会调用其中的 `python` 和 `torchrun`。
每次打开新的 shell 后，执行 `source .venv/bin/activate` 重新激活。

### 3.2 安装 PyTorch 与 OLMo-core

先安装 CUDA 版 PyTorch 和扩展编译工具，再以可编辑模式安装固定版本的 OLMo-core Git 子模块：

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0 torchvision torchaudio

uv pip install 'setuptools<70' wheel packaging ninja
uv pip install --editable 'third_party/OLMo-core[all]'
```

训练命令使用 `third_party/OLMo-core` 中以可编辑模式安装的源码，因此本地源码修改会立即
生效。

### 3.3 安装 attention kernel

先安装与 OLMo-core 配套的固定 FlashAttention-2 版本：

```bash
MAX_JOBS=8 \
uv pip install --no-build-isolation 'flash-attn==2.8.2'
```

默认五阶段配置使用 FlashAttention-3，面向 H100、H800 等 Hopper GPU。请从 OLMo-core 环境采用的
commit 安装 FA3。`FLASH_ATTENTION_FORCE_BUILD=TRUE` 会从该源码构建扩展，`--no-cache` 则避免
复用由其他 feature flags 构建的 wheel：

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTENTION_DISABLE_FP16=TRUE \
FLASH_ATTENTION_DISABLE_SM80=TRUE \
MAX_JOBS=8 \
uv pip install --no-cache --no-build-isolation \
  'flash-attn-3 @ git+https://github.com/Dao-AILab/flash-attention.git@92ca9da8d66f7b34ff50dc080ec0fef9661260d6#subdirectory=hopper'
```

其他受支持的 GPU 使用上述 FA2，并在 `run/run.sh` 的 `all_stage_args` 中加入以下覆盖：

```bash
"--model.block.sequence_mixer.backend=flash_2"
```

默认 HSDP 配方关闭上下文并行。自定义 ring-CP 配置还需安装与 OLMo-core 配套的
`ring-flash-attn` 版本：

```bash
uv pip install --no-build-isolation 'ring-flash-attn==0.1.8'
```

`MAX_JOBS` 应按编译节点的 CPU 和内存调整。

### 3.4 验证训练环境

按本机实际安装的后端执行对应检查：

```bash
uv pip check
# FA2（所有安装）
python -c 'from olmo_core.nn.attention.flash_attn_api import has_flash_attn_2; assert has_flash_attn_2()'
# FA3（仅 Hopper）
python -c 'from olmo_core.nn.attention.flash_attn_api import has_flash_attn_3; assert has_flash_attn_3()'
# ring-flash-attn（仅可选安装）
python -c 'from olmo_core.nn.attention.flash_attn_api import has_ring_flash_attn; assert has_ring_flash_attn()'
python -c 'import dion, torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

## 4. 准备数据集

请按照
[ArchSpace-Collection/OLMo3-1B-Dataset](https://huggingface.co/datasets/ArchSpace-Collection/OLMo3-1B-Dataset)
的 Dataset Card 选择、下载并解压数据；其中也包含空间需求、文件布局、来源和许可信息。

将数据集 `extract.sh` 的输出根目录填入 `run/envs.sh` 的 `olmo3_data_root`。该目录应位于共享
存储，并在所有训练节点上以同一路径可读。

`run.sh` 选中的阶段决定需要准备哪些解压分组：

| `run.sh` 阶段 | 配置 | Dataset Card 分组 | 评测输入 | 从该阶段开始所需的上游 checkpoint |
| ------------- | ---- | ----------------- | --------------- | ------------------------------------ |
| stage 1 | pretraining | stage 1 与 eval | Dolma 2 `tokenizer_json` | — |
| stage 2 | midtraining | stage 2 与 eval | Dolma 2 `tokenizer_json` | stage 1 |
| stage 3 | long-context | stage 3 | — | stage 2 |
| stage 4 | Think SFT | stage 4 | — | stage 3 |
| stage 5 | Instruct SFT | stage 5 | — | stage 4 |

默认启动脚本依次运行五个阶段，因此会使用全部数据分组。缩小阶段循环后，只需准备对应分组。
如果从 stage 2–5 开始，请将前一阶段的 checkpoint 放在默认流水线路径，或通过
`previous_save_folder` 指定。

## 5. 运行训练

### 5.1 配置环境与配方

```bash
cd reproduce/train-olmo-core
cp run/envs.sh.example run/envs.sh
# 编辑 run/envs.sh
```

`.gitignore` 会忽略 `run/envs.sh`。启动前请配置以下机器相关字段：

| 字段 | 含义 |
| ---- | ---- |
| `olmo3_data_root` | 第 4 节所述的数据集解压输出根目录；所有节点使用同一个共享路径。 |
| `out_root` | checkpoint、训练器状态、W&B 文件和数据集缓存的共享持久化根目录。 |
| `tokenizer_json` | 所有节点均可读的 Dolma 2 `tokenizer.json`。启动脚本的必填项，供 stage 1/2 的下游评测器使用。 |
| `wandb_entity`、`wandb_project`、`WANDB_MODE` | W&B 目标与模式，见第 5.3 节。 |

启动脚本接受两个可选位置参数：

```bash
bash run/run.sh [TIMESTAMP] [BASE_PORT]
```

阶段选择、优化器和 Python dotlist 覆盖参数统一在 `run/run.sh` 顶部可编辑的小写变量块中配置：

| `run.sh` 字段 | 用途 |
| ------------ | ---- |
| `pipeline_name` | 决定 `${out_root}/runs/` 下的训练输出名称；启动不同配方时应修改。 |
| `all_stage_args` | 应用于所有选中阶段的 Python CLI 覆盖参数，例如优化器或 attention 后端。 |
| `stage1_args` ... `stage5_args` | 只应用于某一阶段的覆盖参数，例如该阶段的学习率。 |
| `for stage_index in 1 2 3 4 5` | 选择要运行的阶段及其执行顺序。 |

同一个 dotlist 配置项只应放在一个数组中。`all_stage_args` 在各阶段参数之后追加，因此重复项可能覆盖
阶段专用值。优化器变更通常应放进 `all_stage_args`，并在整条 checkpoint 链中保持一致。

例如，只运行 stage 3–5 时，将循环改为：

```bash
for stage_index in 3 4 5; do
```

如果第一个选中阶段是 stage 2–5，则需要对应的上游 checkpoint。默认情况下，stage N 从
`${out_root}/runs/${pipeline_name}/stage(N-1)/checkpoints` 加载。若从该流水线之外的 checkpoint
开始，请在 `run.sh` 对应的 `case` 分支中设置
`previous_save_folder=/path/to/parent/checkpoints`。

如需调整 `data_loader.global_batch_size`，请修改对应阶段的 Python 配置，然后重新 dry-run 整条流水线。
配置会在合并 dotlist 参数之前，根据该值计算临时 checkpoint 间隔。

### 5.2 验证配置

分配完整训练资源前，先解析所有选中配置及覆盖项：

```bash
ENABLE_WANDB=0 DRY_RUN=1 bash run/run.sh 0816_120000 29500
```

`dry-run` 会解析每个选中阶段的配置，包括已有 `_SUCCESS` 标记的阶段，且不会修改训练输出。
正式训练前还需确认数据分片、上游 checkpoint、分布式拓扑和端口。

### 5.3 W&B

`envs.sh.example` 默认 `WANDB_MODE=offline`。这种模式不需要 API 密钥，文件写到每个阶段的
`trainer/wandb/` 下，并自动禁用只能在线工作的远程取消标签。

启动时设置 `ENABLE_WANDB=0` 可关闭 W&B。使用在线记录时，在 `run/envs.sh` 中设置
`WANDB_MODE=online`、`wandb_entity` 和 `wandb_project`，并在调用 shell 中导出
`WANDB_API_KEY`。

时间戳标识 W&B 运行记录，`out_root` 与 `pipeline_name` 标识训练输出。续训时保持
`pipeline_name` 不变，并为每次任务使用新的时间戳，使 W&B 记录彼此独立。同一次多机任务的
所有节点必须使用相同时间戳。

### 5.4 输出与共享缓存

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

`config.json` 是最终生效配置，`data_paths.txt` 记录实际展开的数据文件；复现实验归档时应和 W&B
记录一起保存。配置约每 10 亿 token 写一次临时 checkpoint，只保留一个临时版本，并在阶段
结束时保存最终 checkpoint。

`_SUCCESS` 由 node rank 0 在该阶段 `torchrun` 成功退出后创建，用作进程完成标记。保存步数与
指标应从 checkpoint 和 W&B 记录中确认。

`pipeline_name` 为 `${out_root}/runs/` 下的阶段输出划分命名空间；数据集缓存使用独立路径
`${out_root}/dataset-cache/olmo3-stageN`。当源数据、数据打包方式或配置指纹改变时，请使用
新的 `out_root`，或通过 `run.sh` 中的 `data_work_dir` 指定新的缓存命名空间。

### 5.5 续训与阶段衔接

> **兼容性说明：** 五阶段配方的数据集指纹和数据加载器状态与早期三阶段配方不同。请使用新的
> `pipeline_name` 和缓存命名空间。复用旧阶段输出、缓存或 `_SUCCESS` 标记前，需同时验证模型、
> 优化器、数据和数据加载器状态的兼容性。

在五阶段配方内续训时，保持 `out_root` 和 `pipeline_name` 不变，并使用新的任务时间戳：

```bash
bash run/run.sh 0809_093000
```

启动器和 OLMo-core 的行为是：

1. 训练时若存在 `stageN/_SUCCESS`，启动脚本会跳过该阶段；
2. 否则，当前阶段 `checkpoints/` 中的 checkpoint 会恢复模型、优化器、训练器、数据加载器和
   RNG 状态；
3. stage 2–5 既无标记也无本阶段 checkpoint 时，顶层 `--load_path` 会从前一阶段初始化
   模型和优化器，并从新阶段的 step/epoch 起点开始计数；
4. stage 1 既无标记也无 checkpoint 时从头开始。

因此 stage 2 中断后的重跑会跳过已完成的 stage 1，并从 stage 2 自己的最新 checkpoint 继续；
后续阶段只会在各自上游阶段完成后启动，形成 stage 1 → stage 2 → stage 3 → stage 4 → stage 5
的完整链路。

## 6. 使用 OLMES 评测

训练输出是 OLMo-core 分布式 checkpoint，而 OLMES 的本地模型入口使用 Hugging Face 模型目录。
请先转换 checkpoint，再运行 OLMES。基础模型评测通常使用 stage 3 的最终 checkpoint；对话或
指令遵循评测则转换相应的 stage 4/5 checkpoint，并选择与该阶段匹配的评测套件。做阶段对比时，
应分别转换对应 checkpoint。

### 6.1 转换为 Hugging Face 格式

将 `CHECKPOINT` 设为具体的 `step<N>` 目录：

```bash
export CHECKPOINT=/path/to/out_root/runs/olmo3-1b/stage3/checkpoints/stepNNNNN
export HF_MODEL_DIR=/path/to/out_root/hf/olmo3-1b-stage3-stepNNNNN

python third_party/OLMo-core/src/examples/huggingface/convert_checkpoint_to_hf.py \
  --checkpoint-input-path "${CHECKPOINT}" \
  --huggingface-output-dir "${HF_MODEL_DIR}" \
  --max-sequence-length 32768
```

转换器会从 checkpoint 的 `config.json` 恢复 OLMo 3 架构，并默认使用配置中的
`allenai/dolma2-tokenizer`。离线环境可传入
`--tokenizer /path/to/local/hf-tokenizer-directory`，该路径应是可由
`AutoTokenizer.from_pretrained()` 加载的完整目录。

请保留默认开启的数值验证。转换后可先做最小加载测试：

```bash
python -c 'import os; from transformers import AutoModelForCausalLM, AutoTokenizer; p=os.environ["HF_MODEL_DIR"]; AutoTokenizer.from_pretrained(p); AutoModelForCausalLM.from_pretrained(p); print("HF checkpoint OK")'
```

### 6.2 安装并运行 OLMES

OLMES 需要 PyTorch 2.8，因此使用独立的 uv 虚拟环境。在 `reproduce/train-olmo-core` 中，从固定
commit 安装 OLMES 及其可选 vLLM 依赖：

```bash
uv venv --python 3.12 venv/olmes
uv pip install --python venv/olmes/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 'torch==2.8.0'
uv pip install --python venv/olmes/bin/python \
  'ai2-olmes[gpu] @ git+https://github.com/allenai/olmes.git@5a51f502d463b8cdc4a2dcad7d7096c41ff1197e'
uv pip check --python venv/olmes/bin/python
source venv/olmes/bin/activate
```

Git URL 固定的是 OLMES 源码本身。请将 `uv pip freeze --python venv/olmes/bin/python` 与评测结果
一同保存，以记录实际解析出的间接依赖版本。

stage 3 基础模型的小规模实验可从 OLMo 3 base-easy suites 开始：

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

如 OLMES/vLLM 版本支持该模型，可加 `--model-type vllm` 提高吞吐。正式报告中应保留 OLMES
提交号、完整命令、评测套件、checkpoint step、HF 转换参数和输出目录。`FAST_TASKS`/PPL 用于
训练中的快速评测与监控，最终评测应使用固定版本的 OLMES。对于 stage 4/5 checkpoint，请选择并
记录合适的对话或指令遵循评测套件。
