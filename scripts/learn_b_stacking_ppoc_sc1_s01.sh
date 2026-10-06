source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/conda/envs/maniskill3_env

# Run multiple seeds
for seed in 0 1
do
    echo "Running seed ${seed}..."
    python3 src/skill_refactor/main.py --seed ${seed} --approach rl_planning_states --env blocked_stacking \
        --exp_name skill_1127_sc1_pred_seed${seed} \
        --lll_config config/lifelong_learning/blocked_stacking_sc1.yaml \
        --rl_algo PPOC \
        --control_mode pd_joint_delta_pos \
        --pre_trained_policy_path trained_policies/runs/skill_1127_sc1_seed${seed}/best_ppo_ckpt.pt \
        --pred_net_save_dir skill_1127_sc1_pred_nets_seed${seed} \
        --log_file logs/skill_1127_sc1_pred_seed${seed}.log
done
