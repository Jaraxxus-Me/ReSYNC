source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/conda/envs/maniskill3_env

# Run multiple seeds
for seed in 0
do
    echo "Running seed ${seed}..."
    python3 src/skill_refactor/main.py --seed ${seed} --approach rl_planning_states --env cluttered_drawer \
        --exp_name skill_pred_0103_c_drawer_sc12_seed${seed} \
        --lll_config config/lifelong_learning/cluttered_drawer_sc12_2_seed${seed}.yaml \
        --rl_algo PPOC \
        --dreaming_noise_base_var 0.0 \
        --delta_finger_control False \
        --control_mode pd_joint_delta_pos \
        --log_file logs/skill_pred_0109_c_drawer_sc12_seed${seed}.log
done
