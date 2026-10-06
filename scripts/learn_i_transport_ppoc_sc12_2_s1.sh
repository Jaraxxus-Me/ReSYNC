source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/conda/envs/maniskill3_env

# Run multiple seeds
for seed in 1
do
    echo "Running seed ${seed}..."
    python3 src/skill_refactor/main.py --seed ${seed} --approach rl_planning_states --env icy_transport \
        --exp_name skill_0101_i_transport_sc12_2_seed${seed} \
        --lll_config config/lifelong_learning/icy_transport_sc12_2_seed${seed}.yaml \
        --rl_algo PPOC \
        --control_mode force_torque \
        --log_file logs/skill_planner_data_0101_i_transport_sc12_2_seed${seed}.log
done
