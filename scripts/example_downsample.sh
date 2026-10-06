#!/bin/bash
# Example script for downsampling a planner dataset

# Activate virtual environment
source .venv/bin/activate

# Example: Downsample a dataset keeping 5 intermediate states per operator segment
# python scripts/downsample_planner_dataset.py \
#     --input_path training_data/cluttered_room/Planner_data/scenario1/seed_2_full \
#     --output_path training_data/cluttered_room/Planner_data/scenario1/seed_2 \
#     --num_intermediate_samples 5

# python scripts/downsample_planner_dataset.py \
#     --input_path training_data/cluttered_room/Planner_data/scenario1/seed_4_full \
#     --output_path training_data/cluttered_room/Planner_data/scenario1/seed_4 \
#     --num_intermediate_samples 5

python scripts/downsample_planner_dataset.py \
    --input_path training_data/cluttered_room/Planner_data/scenario1/seed_2_full \
    --output_path training_data/cluttered_room/Planner_data/scenario1/seed_2 \
    --num_intermediate_samples 5

# Example: Minimal downsampling - only keep operator boundaries (no intermediate states)
# python scripts/downsample_planner_dataset.py \
#     --input_path results/cluttered_table/lifelong_ref/seed0/planner_dataset \
#     --output_path results/cluttered_table/lifelong_ref/seed0/planner_dataset_minimal \
#     --num_intermediate_samples 0

# Example: Load only first 100 trajectories
# python scripts/downsample_planner_dataset.py \
#     --input_path results/cluttered_table/lifelong_ref/seed0/planner_dataset \
#     --output_path results/cluttered_table/lifelong_ref/seed0/planner_dataset_downsampled \
#     --num_intermediate_samples 5 \
#     --num_traj 100
