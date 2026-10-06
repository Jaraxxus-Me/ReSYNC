source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/conda/envs/maniskill3_env

# Run multiple seeds
for seed in 3
do
    echo "Running seed ${seed}..."
    python3 src/skill_refactor/main.py --seed ${seed} --approach rl_planning_states --env icy_transport \
        --exp_name skill_pred_search_0101_i_transport_sc1_seed${seed} \
        --lll_config config/lifelong_learning/icy_transport_sc1.yaml \
        --rl_algo PPOC \
        --control_mode force_torque \
        --pre_trained_policy_path trained_policies/runs/skill_0101_i_transport_sc1_seed${seed}/best_ppo_ckpt.pt \
        --pred_net_save_dir skill_0101_i_transport_sc1_pred_nets_seed${seed} \
        --log_file logs/skill_pred_search_0101_i_transport_sc1_seed${seed}.log
done
