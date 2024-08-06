#!/bin/bash

# e.g. bash all-seeds.sh stack_nm mt feb17 sqil --exp_name feb17
# e.g. bash all-seeds.sh stack_nm st feb17 sqil --exp_name feb17
# e.g. bash all-seeds.sh stack_nm mt feb17 rce --exp_name feb17
# e.g. bash all-seeds.sh stack_nm mt feb17 discriminator --exp_name feb17

seeds=(1 2 3 4 5)
# seeds=(1)
main_task="$1"
task_mode="$2"  # st or mt for single or multitask
job_name_post="$3"
algo="$4"

if [ "${main_task}" = "stack" ] || [ "${main_task}" = "bring" ] || [ "${main_task}" = "stack_nm" ] || [ "${main_task}" = "bring_nm" ]; then
    if [ ${task_mode} = "st" ]; then
        req_time="00-12:00"
    else
        req_time="00-12:00"
    fi
elif [ "${main_task}" = "unstack" ] || [ "${main_task}" = "unstack_nm" ]; then
    if [ ${task_mode} = "st" ]; then
        req_time="00-18:00"
    else
        req_time="00-24:00"
    fi
elif [ "${main_task}" = "insert_nb" ] || [ "${main_task}" = "insert_nb_nm" ]; then
    if [ ${task_mode} = "st" ]; then
        req_time="00-18:00"
    else
        req_time="00-24:00"
    fi
else
    echo "Invalid MAIN_TASK ${main_task}"
    exit 1
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
            --job-name="${seed}-${main_task}-${task_mode}-${algo}-${job_name_post}" \
            --time="${req_time}" \
            "single_job.sh" \
            --seed "${seed}" \
            --main_task "${main_task}" \
            --reward_model "${algo}" \
            --single_task \
            "$@"
    else
        sbatch \
            --job-name="${seed}-${main_task}-${task_mode}-${algo}-${job_name_post}" \
            --time="${req_time}" \
            "single_job.sh" \
            --seed "${seed}" \
            --main_task "${main_task}" \
            --reward_model "${algo}" \
            "$@"
    fi
done