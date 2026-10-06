source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/conda/envs/maniskill3_env

# Run multiple seeds
for seed in 3 4
do
    echo "Running seed ${seed}..."
    python3 src/skill_refactor/main.py --seed ${seed} --approach rl_planning_states --env blocked_stacking \
        --exp_name skill_1130_sc123_3_seed${seed} \
        --lll_config config/lifelong_learning/blocked_stacking_sc123_3_seed${seed}.yaml \
        --rl_algo PPOC \
        --control_mode pd_joint_delta_pos \
        --pre_trained_policy_path trained_policies/runs/skill_1128_sc123_3_seed${seed}/best_ppo_ckpt.pt \
        --log_file logs/skill_1130_sc123_3_seed${seed}.log
done
