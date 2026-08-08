# OLMo 3 1B 三阶段复现

[English](README.md) | 中文

本目录提供 OLMo 3 1B 的三阶段训练配方与启动脚本：

1. stage 1：pretraining；
2. stage 2：midtraining；
3. stage 3：long-context extension。

模型实现、分布式训练器、checkpoint I/O 和数据集实现均来自 OLMo-core。本目录只保存某次
复现所需的配置和运行入口，以便把“实验配方”与“通用训练框架”隔离开：修改本目录中的配方
不会污染 OLMo-core，升级或修复训练框架也不需要把整个框架复制进本仓库。

本复现必须使用以下自定义 OLMo-core 源码分支，而不是 PyPI 上的通用版本：

- [https://github.com/JT-Ushio/OLMo-core-muon-fix/tree/ready_for_archspace_base](https://github.com/JT-Ushio/OLMo-core-muon-fix/tree/ready_for_archspace_base)

该分支包含本配方所依赖的 Muon 修复。OLMo 3 模型本身已由
`TransformerConfig.olmo3_1B()` 实现，因此这里没有额外的 modeling 文件。

## 1. 目录结构

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

## 2. 配方概览


| 阶段    | 数据 mix                         | 序列长度 | 全局 batch（token） | 并行方式 | 默认 Muon LR |
| --------- | ---------------------------------- | ---------: | --------------------: | ---------- | -------------: |
| stage 1 | `OLMo_mix_0625_150Bsample`       |    4,096 |           2,097,152 | HSDP     |       `1e-3` |
| stage 2 | `OLMo_midtraining_mix_0625_100B` |    4,096 |           2,097,152 | HSDP     |       `5e-4` |
| stage 3 | `OLMo_longmino_mix_0625`         |   65,536 |           4,194,304 | HSDP     |       `5e-4` |

三个阶段默认都使用 BF16、FlashAttention-3 和 Muon，并各自训练一个完整数据
epoch。stage 1/2 使用固定长度数据集；stage 3 使用 document packing、文档内 attention
mask 和 8 倍 YaRN RoPE scaling。三个阶段均使用 HSDP，不使用 context parallelism。
默认的 FlashAttention-3 配置面向 Hopper GPU；其他支持的 GPU 应按 3.2 节切换到 FlashAttention-2。

也可以用 `adam` 选择 SkipStep AdamW 配方，但同一 pipeline 的三个阶段及所有 resume 必须使用
同一种 optimizer，因为后续阶段会继承前一阶段的 optimizer state。

这些是从 OLMo 3 7B 官方配方缩放到 1B 模型的实验配置，并不是官方发布、已调优的 OLMo 3
1B recipe。

## 3. 环境安装

本项目采用 PyTorch 2.10 和 CUDA 12.8，但原则上也可使用任何与 OLMo-core 及其他依赖兼容的版本。

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0 torchvision torchaudio
```

### 3.1 从源码安装自定义 OLMo-core

推荐保留一个独立源码 checkout，并以 editable 方式安装：

```bash
export OLMO_CORE_SRC=/path/to/OLMo-core-muon-fix
git clone --branch ready_for_archspace_base --single-branch \
  https://github.com/JT-Ushio/OLMo-core-muon-fix.git "${OLMO_CORE_SRC}"

pip install -e "${OLMO_CORE_SRC}[all]"
```

### 3.2 安装 attention kernels

本节的 attention kernels 都直接安装最新版，不固定 tag、commit 或 package
version。先安装 FlashAttention-2：

```bash
MAX_JOBS=8 python -m pip install --upgrade --no-build-isolation flash-attn
```

Hopper GPU（例如 H100/H800）可以使用 FlashAttention-3，从 FlashAttention 默认分支的
`hopper/` 目录安装：

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

其他支持的 GPU 使用 FlashAttention-2。三个 cfg 从上游同步的默认 backend 都是
`flash_3`；使用 FA2 时，在 `run/run.sh` 的 `extra_args` 数组中加入以下覆盖，使它同时
作用于三个阶段：

```bash
"--model.attn_backend=flash_2"
```

`ring-flash-attn` 作为可选 backend 保留。当自定义配置启用 ring context parallelism 时
再安装；当前 HSDP、无 CP 的三阶段配方不需要它：

```bash
pip install ring-flash-attn
```

`MAX_JOBS` 应按编译节点的 CPU 和内存调整。可按实际选择的 backend 分别验证：

```bash
python -m pip check
# FA2（所有安装）
python -c 'from olmo_core.nn.attention.flash_attn_api import has_flash_attn_2; assert has_flash_attn_2()'
# FA3（仅 Hopper）
python -c 'from olmo_core.nn.attention.flash_attn_api import has_flash_attn_3; assert has_flash_attn_3()'
# ring-flash-attn（仅可选安装）
python -c 'from olmo_core.nn.attention.flash_attn_api import has_ring_flash_attn; assert has_ring_flash_attn()'
python -c 'import dion, torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

## 4. 数据准备

### 4.1 数据格式

配置直接使用安装在 OLMo-core 包中的四份 `DataMix` manifest。它们列出的每个 `.npy` 路径是
OLMo-core 约定的、可由 `numpy.memmap` 读取的一维 token-ID 二进制数组，而不是任意文本文件，
也不能只靠把文件改名为 `.npy` 得到。Dolma 2 tokenizer 的词表大小为 100,278，因此本配方会
推断数组 dtype 为 `uint32`。不同文档需要以 EOS token（ID `100257`）正确分隔，stage 3 的
document packing 和文档内 mask 依赖这些边界。

数据来自olmo官方，本仓库提供 Huggingface 再发布版本：链接TODO

### 4.2 `olmo3_data_root` 的预期布局

`run.sh` 把 `envs.sh` 中的 `olmo3_data_root` 原样传给三个配置。OLMo-core 再把它作为 manifest
内所有相对路径的前缀。大致目录如下；省略号代表 manifest 中的全部 source 和 shard：

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

stage 1 使用第一棵树，stage 2 使用第二棵树，stage 3 使用第三棵树；stage 1/2 的 in-loop LM
evaluation 都需要最后一棵 validation 树。manifest 文件名必须逐项匹配，不能只提供相似的
顶层目录。

`tokenizer_json` 是另一项必填路径：它应指向所有节点都能读取的 Dolma 2 `tokenizer.json`，
供 stage 1/2 的 in-loop downstream evaluator 使用。它不替代上述已分词训练数组。

## 5. 运行训练

```bash
cd reproduce/olmo-core-backend
cp run/envs.sh.example run/envs.sh
# edit run/envs.sh
bash run/run.sh
```

`run/envs.sh` 已被 `.gitignore` 排除。

### 5.1 W&B

`envs.sh.example` 默认 `WANDB_MODE=offline`。这种模式不需要 API key，文件写到每个阶段的
`trainer/wandb/` 下，并自动禁用只能在线工作的 remote cancel tags。

timestamp 只参与 W&B run name/ID。resume 同一个
实验时保持 `pipeline_name` 不变，但每次重新发起任务建议给一个新 timestamp，避免新的 W&B
片段覆盖或混入上一次 attempt。多机同一次 attempt 必须使用同一个 timestamp。

### 5.2 输出目录

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

`config.json` 是最终生效配置，`data_paths.txt` 记录实际展开的数据文件；复现实验归档时应和 W&B
记录一起保存。配置约每 10 亿 token 写一次临时 checkpoint，只保留一个临时版本，并在阶段
结束时保存最终 checkpoint。

`_SUCCESS` 由 node rank 0 在该阶段 `torchrun` 成功退出之后创建。它表示进程成功完成，不会
再次检查 checkpoint step 或指标内容。

### 5.3 Resume 与阶段衔接

正常 resume 不需要手工指定 checkpoint：

```bash
# out_root、pipeline_name 保持不变；使用新的 attempt timestamp。
bash run/run.sh 0809_093000
```

启动器和 OLMo-core 的行为是：

1. 存在 `stageN/_SUCCESS`：直接跳过该阶段；
2. 不存在 `_SUCCESS`，但当前阶段 `checkpoints/` 中有 checkpoint：恢复该阶段的 model、
   optimizer、trainer、data-loader 和 RNG 状态；
3. 当前 stage 2/3 没有 checkpoint：通过顶层 `--load_path` 从前一阶段 checkpoint 初始化 model
   和 optimizer，但不继承前一阶段的 step/epoch 进度；
4. 当前 stage 1 没有 checkpoint：从头开始。

因此 stage 2 中断后的重跑会跳过已完成的 stage 1，并从 stage 2 自己的最新 checkpoint 继续；
stage 2 完成后才进入 stage 3。

## 6. 使用 OLMES 评测

训练输出是 OLMo-core distributed checkpoint，而 OLMES 的本地模型入口使用 Hugging Face
模型目录。因此先转换 checkpoint，再运行 OLMES。通常评测 stage 3 的最终 checkpoint；若要画
阶段对比，则分别转换 stage 1/2/3。

### 6.1 转换为 Hugging Face 格式

选中具体的 `step<N>` 目录，而不是它的 `checkpoints/` 父目录：

```bash
export CHECKPOINT=/path/to/out_root/runs/olmo3-1b/stage3/checkpoints/step11921
export HF_MODEL_DIR=/path/to/out_root/hf/olmo3-1b-stage3-step11921

python "${OLMO_CORE_SRC}/src/examples/huggingface/convert_checkpoint_to_hf.py" \
  --checkpoint-input-path "${CHECKPOINT}" \
  --huggingface-output-dir "${HF_MODEL_DIR}" \
  --max-sequence-length 65536
```

转换器会从 checkpoint 的 `config.json` 恢复 OLMo 3 架构，并默认使用配置中的
`allenai/dolma2-tokenizer`。离线环境可额外传
`--tokenizer /path/to/local/hf-tokenizer-directory`；这里应给一个可由
`AutoTokenizer.from_pretrained()` 加载的完整目录，而不是单独的 `tokenizer.json`。

默认转换包含数值验证。除非已经单独验证且明确接受风险，不建议使用 `--skip-validation`。
转换后可先做最小加载测试：

```bash
python -c 'import os; from transformers import AutoModelForCausalLM, AutoTokenizer; p=os.environ["HF_MODEL_DIR"]; AutoTokenizer.from_pretrained(p); AutoModelForCausalLM.from_pretrained(p); print("HF checkpoint OK")'
```

### 6.2 安装并运行 OLMES

建议为评测创建独立环境，避免 vLLM/Transformers 版本反向影响训练环境：

```bash
git clone https://github.com/allenai/olmes.git /path/to/olmes
cd /path/to/olmes
python -m pip install -e '.[gpu]'
git rev-parse HEAD
```

小规模实验可从 OLMo 3 base-easy suites 开始：

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
commit、完整命令、task suite、checkpoint step、HF 转换参数和输出目录。训练配置内置的
`FAST_TASKS`/PPL in-loop evaluation 用于训练监控，不能替代最终、版本固定的 OLMES 评测。
