source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/conda/envs/maniskill3_env

# Run multiple seeds
for seed in 2
do
    echo "Running seed ${seed}..."
    python3 src/skill_refactor/main.py --seed ${seed} --approach rl_planning_states --env cluttered_drawer \
        --exp_name skill_pred_0103_c_drawer_sc1_seed${seed} \
        --lll_config config/lifelong_learning/cluttered_drawer_sc1.yaml \
        --rl_algo PPOC \
        --dreaming_noise_base_var 0.0 \
        --delta_finger_control False \
        --control_mode pd_joint_delta_pos \
        --pre_trained_policy_path trained_policies/runs/skill_1231_sc1_seed0/best_ppo_ckpt.pt \
        --pred_net_save_dir c_drawer_sc1_pred_nets_seed${seed} \
        --log_file logs/skill_pred_0103_c_drawer_sc1_seed${seed}.log
done
