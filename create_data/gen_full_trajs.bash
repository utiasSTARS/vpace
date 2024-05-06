#!/bin/bash

# selecting stack_0 will additionally generate both open and close data

TASK=$1  # task must be full env name, e.g. stack_0, bring_0, insert_no_bring_0, etc. -- corresponding open data also made
NUM_STEPS=$2
default_expert_dir="${VPACE_TOP_DIR}experts"
EXPERT_DIR=${3:-${default_expert_dir}}
default_save_dir="./expert-data/${TASK}_${NUM_STEPS}_steps/"
SAVE_DIR=${4:-${default_save_dir}}
scheduler_period=32

model_file="${EXPERT_DIR}/${TASK}.pt"
config_file="${EXPERT_DIR}/${TASK}_setting.pkl"

echo "Generating ${NUM_STEPS} data for ${TASK}, expert model from ${model_file}, saving at ${SAVE_DIR}"

if [ "${TASK}" = "stack_0" ]; then
    forced_schedule="{0: {0: ([0, 1, 2, 3, 4, 5], [.1, .1, .5, .1, .1, .1], ['k', 'd', 'd', 'd', 'd', 'd']), 16: 0}, 1: {0: 3, 7: 1}}"
    aux_override="0,1,2"

elif [ "${TASK}" = "reach_0" ]; then
    forced_schedule=""
    aux_override="1"

elif [ "${TASK}" = "lift_0" ]; then
    forced_schedule=""
    aux_override="2"

elif [ "${TASK}" = "move_obj_0" ]; then
    forced_schedule=""
    aux_override="4"

elif [ "${TASK}" = "unstack_stack_env_only_0" ]; then
    forced_schedule="{0: {0: ([0, 1, 2, 3, 4, 5], [.1, .1, .5, .1, .1, .1], ['k', 'd', 'd', 'd', 'd', 'd']), 16: 0}, 1: {0: 3, 7: 1}}"
    # no aux override because we need data from all tasks for unstack

elif [ "${TASK}" = "bring_0" ] || [ "${TASK}" = "insert_no_bring_0" ]; then
    forced_schedule="{0: {0: ([0, 1, 2, 3, 4, 5], [.1, .1, .5, .1, .1, .1], ['k', 'd', 'd', 'd', 'd', 'd']), 16: 0}, 1: {0: 3, 7: 1}}"
    # num_steps_per_buffer="${NUM_STEPS}, ${NUM_STEPS}, ${NUM_STEPS}, 0, 0, 0"
    aux_override="0,2"

fi


python -m rl_sandbox.examples.lfgp.experts.create_expert_data \
    --model_path="${model_file}" \
    --config_path="${config_file}" \
    --save_path="${SAVE_DIR}" \
    --num_episodes=10000000 \
    --num_steps="${NUM_STEPS}" \
    --seed=1 \
    --forced_schedule="${forced_schedule}" \
    --scheduler_period="${scheduler_period}" \
    --success_only \
    --reset_on_success \
    --reset_between_intentions \
    --aux_override="${aux_override}"\
    --render
