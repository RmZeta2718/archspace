#!/usr/bin/env bash
{
    set -euo pipefail

    script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    reproduce_dir=$(cd -- "${script_dir}/.." && pwd)
    env_file=${script_dir}/envs.sh
    [[ -f "${env_file}" ]] || {
        echo "Local environment file not found. Copy ${script_dir}/envs.sh.example to ${env_file}." >&2
        exit 1
    }
    # shellcheck source=/dev/null
    source "${env_file}"

    # All nodes in one distributed attempt must receive the same timestamp and
    # base port. Give every resume attempt a new timestamp so its W&B ID differs.
    timestamp=${1:-$(date +'%m%d_%H%M%S')}
    base_port=${2:-29500}
    pipeline_name=olmo3-1b
    config_basename=${reproduce_dir}/cfgs/OLMo3-1B
    extra_args=(
        # For a short smoke run, uncomment both overrides. Do not use them for
        # the full reproduction.
        # "--trainer.max_duration.value=10"
        # "--trainer.max_duration.unit=steps"
    )

    olmo3_data_root=${olmo3_data_root:?Set olmo3_data_root in envs.sh}
    out_root=${out_root:?Set out_root in envs.sh}
    tokenizer_json=${tokenizer_json:?Set tokenizer_json in envs.sh}
    pipeline_root=${out_root}/runs/${pipeline_name}

    for stage_index in 1 2 3; do
        stage="stage${stage_index}"
        previous_save_folder=
        stage_args=()

        case "${stage}" in
            stage1)
                config_file=${config_basename}-pretrain.py
                stage_args=(
                    "--trainer.callbacks.downstream_evaluator.tokenizer.identifier=${tokenizer_json}"
                )
                ;;
            stage2)
                config_file=${config_basename}-midtraining.py
                stage_args=(
                    "--trainer.callbacks.downstream_evaluator.tokenizer.identifier=${tokenizer_json}"
                )
                previous_save_folder=${pipeline_root}/stage1/checkpoints
                ;;
            stage3)
                config_file=${config_basename}-long-context.py
                previous_save_folder=${pipeline_root}/stage2/checkpoints
                ;;
        esac

        stage_port=$((base_port + stage_index))
        run_name=${pipeline_name}-${stage}
        run_root=${pipeline_root}/${stage}
        data_work_dir=${out_root}/dataset-cache/olmo3-${stage}
        save_folder=${run_root}/checkpoints
        success_marker=${run_root}/_SUCCESS
        if [[ "${DRY_RUN:-0}" != "1" && -f "${success_marker}" ]]; then
            echo "Skipping ${stage}; success marker already exists at '${success_marker}'"
            continue
        fi

        train_args=(
            "--name=${run_name}"
            "--data-root=${olmo3_data_root}"
            "--save-folder=${save_folder}"
            "--work-dir=${data_work_dir}"
            "--trainer.work_dir=${run_root}/trainer"
        )
        if [[ -n "${previous_save_folder}" ]]; then
            # On a fresh stage this initializes model and optimizer state from the
            # parent stage without loading parent trainer progress. If the current
            # stage already has a checkpoint, OLMo-core resumes full local state.
            train_args+=("--load_path=${previous_save_folder}")
        fi

        if [[ "${ENABLE_WANDB:-1}" == "1" ]]; then
            train_args+=(
                "--trainer.callbacks.wandb.enabled=true"
                "--trainer.callbacks.wandb.entity=${wandb_entity:?Set wandb_entity in envs.sh}"
                "--trainer.callbacks.wandb.project=${wandb_project:?Set wandb_project in envs.sh}"
                "--trainer.callbacks.wandb.group=${pipeline_name}"
                "--trainer.callbacks.wandb.name=${run_name}_${timestamp}"  # wandb.id = wandb.name
            )
            if [[ "${WANDB_MODE:-online}" != "offline" ]]; then
                : "${WANDB_API_KEY:?Export WANDB_API_KEY before launching for online W&B logging}"
                export WANDB_API_KEY
            else
                # Remote cancel tags cannot be observed by an offline W&B run.
                train_args+=("--trainer.callbacks.wandb.cancel_tags=null")
            fi
        fi
        train_args+=(
            "${stage_args[@]}"
            "${extra_args[@]}"
        )

        echo "Starting ${stage} for pipeline '${pipeline_name}' on port ${stage_port}"
        if [[ "${DRY_RUN:-0}" == "1" ]]; then
            python "${config_file}" --dry-run "${train_args[@]}"
            continue
        fi

        if [[ "${NNODES:-1}" == "1" ]]; then
            torchrun_args=(--standalone "--nproc-per-node=${NPROC_PER_NODE:-gpu}")
        else
            torchrun_args=(
                "--nnodes=${NNODES}"
                "--node-rank=${NODE_RANK}"
                "--nproc-per-node=${NPROC_PER_NODE}"
                "--master-addr=${MASTER_ADDR}"
                "--master-port=${stage_port}"
            )
        fi
        mkdir -p "${run_root}"
        torchrun "${torchrun_args[@]}" "${config_file}" -- "${train_args[@]}"

        [[ "${NODE_RANK:-0}" == "0" ]] && touch "${success_marker}"
    done

    echo "Pipeline '${pipeline_name}' completed all three stages"
    exit
}
