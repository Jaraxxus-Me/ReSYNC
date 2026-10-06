source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/conda/envs/maniskill3_env

# Run multiple seeds
for seed in 0
do
    echo "Running seed ${seed}..."
    python3 src/skill_refactor/main.py --seed ${seed} --approach rl_planning_states --env blocked_stacking \
        --exp_name skill_1206_sc12_2_seed${seed} \
        --lll_config config/lifelong_learning/blocked_stacking_sc12_2_seed${seed}.yaml \
        --rl_algo PPOC \
        --control_mode pd_joint_delta_pos \
        --log_file logs/skill_pred_1206_sc12_2_seed${seed}.log
done
