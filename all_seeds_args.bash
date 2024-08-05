#!/bin/bash

# Code for running multiple seeds and configurations in series

# e.g. bash all_seeds_args.bash --main_task lift --device cuda:0 --exp_name feb20 --reward_model sqil

seeds=(1 2 3 4 5)

echo "Starting for loop of execution with args $@"
for seed in "${seeds[@]}"; do
    echo "Running seed ${seed}, args $@"

    python run_vpace.py \
        --seed "${seed}" \
        "$@"
done