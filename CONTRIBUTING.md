# Contributing to ArchSpace

ArchSpace collects focused LLM architecture innovations in separate `arch/<name>` branches. Each
contribution implements one Architecture Proposal and publishes the code and experiment records needed
to reproduce it.

## Quick start

1. Open an [Architecture Proposal Issue](https://github.com/InternLM/archspace/issues/new/choose) and
   discuss the proposal with the community.
2. Once the discussion supports moving forward, maintainers mark the Issue `in-progress` and create its
   upstream `arch/<proposal-name>` branch from `arch/template` as the eventual PR target.
3. Fork the repository and create a work branch from whichever existing branch is the most useful
   implementation base.
4. Implement the architecture and its training and evaluation workflows, run the relevant tests and
   experiments, complete the root README, and archive the results on W&B.
5. Open a PR into the upstream `arch/<proposal-name>` branch and complete the PR template provided by
   GitHub.
6. After the implementation is merged, maintainers label the proposal `verified` and close its Issue.

## Choose an implementation base

Development may start from any branch that helps implement the proposal:

- [`arch/base`](https://github.com/InternLM/archspace/tree/arch/base) provides a baseline OLMo-core
  reproduction.
- [`arch/morphnorm`](https://github.com/InternLM/archspace/tree/arch/morphnorm) shows an architecture
  extension integrated with a customized training system and kernel.
- [`arch/template`](https://github.com/InternLM/archspace/tree/arch/template) is a blank starting point.

The upstream proposal branch created by maintainers is initially based on `arch/template`, but your work
branch does not have to be. It only needs to be mergeable into that proposal branch. Keep the proposal,
motivation, parent architecture, and validation plan in the Issue; the branch README links the Issue and
focuses on reproduction.

## Organize the reproduction

An architecture branch separates project-owned architecture and reproduction code from general
infrastructure source:

- `archs/<architecture>/`: the portable Hugging Face Transformers implementation;
- `reproduce/train/` and `reproduce/eval/`: project-owned configurations, launch entry points, adapters
  or extensions, and tests;
- `reproduce/<workflow-directory>/third_party/`: external training, evaluation, or kernel repositories,
  preferably recorded as Git submodules so their own history and exact revision remain intact;
- `README.md`: the step-by-step reproduction instructions and result links.

This boundary keeps the architecture-specific change visible while allowing infrastructure such as
OLMo-core, Megatron-LM, Open Instruct, evaluation harnesses, and customized kernels to evolve in their
own repositories. Do not copy an infrastructure source tree into ArchSpace after removing its `.git`
directory.

Organize `reproduce/` by **workflow**:

- `train`: all training phases used by the proposal, including pre-training, mid-training,
  long-context training, supervised fine-tuning, and instruction tuning;
- `eval`: evaluation and result aggregation.

The recommended layout is:

```text
README.md
archs/
└── <architecture>/
    ├── __init__.py
    ├── configuration_<architecture>.py
    └── modeling_<architecture>.py
reproduce/
├── train/
│   ├── cfgs/
│   ├── run/
│   ├── src/
│   ├── tests/
│   └── third_party/
└── eval/
    ├── cfgs/
    ├── run/
    ├── src/
    ├── tests/
    └── third_party/
```

Each workflow owns its configurations, launch code, workflow-specific adapters, tests, and external
source references. Shared portable model code remains in `archs/`.

When one workflow has independent implementations for more than one infrastructure system, suffix the
workflow name with the system, for example `train-olmo-core/` and `train-megatron-lm/`. Omit the system
suffix when a workflow has only one implementation.

### Manage infrastructure repositories

Prefer adding source dependencies as Git submodules under the owning workflow's `third_party/` directory:

```bash
git submodule add <repository-url> \
  reproduce/<workflow-directory>/third_party/<repository-name>
git -C reproduce/<workflow-directory>/third_party/<repository-name> checkout <tested-commit>
git add .gitmodules reproduce/<workflow-directory>/third_party/<repository-name>
```

If a submodule is impractical, record the accessible repository URL and exact tested commit in the
README, together with clone, checkout, build, and install commands. A branch name alone is not a stable
revision.

When architecture-specific and infrastructure changes cannot be separated cleanly, they may remain
together in an infrastructure fork under the workflow's `third_party/` directory. The portable
Transformers implementation is still maintained in `archs/`, and the README points to the corresponding
implementation inside the fork.

Commit and push infrastructure changes before updating the submodule pointer or README revision. Never
replace an external repository with a copied source tree whose `.git` history has been removed.

Clone a contribution with its submodules using:

```bash
git clone --recurse-submodules --branch arch/<proposal-name> <archspace-repository-url>

# In an existing checkout:
git submodule update --init --recursive
```

## Implement, test, and document the reproduction

`arch/base` normally serves as the comparison baseline for a new architecture. Unless the architecture
itself requires a different setting, keep the dataset, tokenizer, token budget, batch size, optimizer
and schedule, sequence length, precision, and evaluation protocol as close as practical to its defaults.
Aligning these non-architectural settings isolates the architecture change and supports a fair
comparison. Keep unavoidable differences minimal and record their effective values and reasons in the
root README.

Run the focused tests and smoke commands for every changed workflow, followed by the experiments agreed in
the Architecture Proposal Issue. Keep enough effective configuration in W&B and the root README to
reproduce each run and compare it with its baseline.

Follow the root README template. It should contain only operational reproduction information: the actual
directory layout, environment and dependencies, inputs, commands, outputs, W&B records, artifacts, and
troubleshooting notes. Link the proposal Issue instead of repeating its design rationale or experiment
plan.

Keep datasets, model weights, checkpoints, logs, caches, build products, credentials, and machine-local
configuration outside Git. Preserve source attribution and licenses for adapted code.
