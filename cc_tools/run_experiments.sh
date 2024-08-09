#!/bin/bash

# e.g. bash run_experiments.sh stack_nm mt feb17 sqil
# e.g. bash run_experiments.sh stack_nm st feb17 sqil
# e.g. bash run_experiments.sh stack_nm mt feb17 rce
# e.g. bash run_experiments.sh stack_nm mt feb17 discriminator

seeds=(1 2 3 4 5)
# seeds=(1)
main_task="$1"
task_mode="$2"  # st or mt for single or multitask
exp_name="$3"
algo="$4"

if [ "${main_task}" = "lift" ] ||[ "${main_task}" = "move" ] ||[ "${main_task}" = "reach" ] ||[ "${main_task}" = "stack" ] || [ "${main_task}" = "bring" ] || [ "${main_task}" = "stack_nm" ] || [ "${main_task}" = "bring_nm" ]; then
    if [ ${task_mode} = "st" ]; then
        req_time="00-12:00"
    else
        req_time="00-12:00"
    fi
    env_type="manipulator_learning"
    env_name="PandaPlayInsertTrayXYZState"
elif [ "${main_task}" = "unstack" ] || [ "${main_task}" = "unstack_nm" ]; then
    if [ ${task_mode} = "st" ]; then
        req_time="00-18:00"
    else
        req_time="00-24:00"
    fi
    env_type="manipulator_learning"
    env_name="PandaPlayInsertTrayXYZState"
elif [ "${main_task}" = "insert_nb" ] || [ "${main_task}" = "insert_nb_nm" ]; then
    if [ ${task_mode} = "st" ]; then
        req_time="00-18:00"
    else
        req_time="00-24:00"
    fi
    env_type="manipulator_learning"
    env_name="PandaPlayInsertTrayXYZState"
else
    env_name="${main_task}"

    if [[ "${env_name}" = *"dp"* ]] || [[ "${env_name}" = "relocate-human-v0" ]]; then
        if [ ${task_mode} = "st" ]; then
            req_time="00-24:00"  # for 1.5M steps
        else
            req_time="00-28:00"  # for 1.5M steps
        fi
    else
        req_time="00-12:00"  # for 500k or 300k steps
    fi

    if [[ "${env_name}" = *"sawyer"* ]]; then
        env_type="sawyer"
    elif [[ "${env_name}" = *"human"* ]]; then
        env_type="hand_dapg"
    else
        echo "Invalid MAIN_TASK ${main_task}"
        exit 1
    fi
fi

# remove first 4 args so we can still pass python args
shift;
shift;
shift;
shift;

echo "Submitting 5 jobs for main task ${main_task}, task mode ${task_mode}, algo ${algo}, req time: ${req_time}"
for seed in "${seeds[@]}"
do
    if [ ${task_mode} = "st" ]; then
        sbatch \
            --job-name="${seed}-${main_task}-${task_mode}-${algo}-${exp_name}" \
            --time="${req_time}" \
            "single_job.sh" \
            --exp_name "${exp_name}" \
            --seed "${seed}" \
            --env_type "${env_type}" \
            --env_name "${env_name}" \
            --main_task "${main_task}" \
            --reward_model "${algo}" \
            --single_task \
            "$@"
    else
        sbatch \
            --job-name="${seed}-${main_task}-${task_mode}-${algo}-${exp_name}" \
            --time="${req_time}" \
            "single_job.sh" \
            --exp_name "${exp_name}" \
            --seed "${seed}" \
            --env_type "${env_type}" \
            --env_name "${env_name}" \
            --main_task "${main_task}" \
            --reward_model "${algo}" \
            "$@"
    fi
done