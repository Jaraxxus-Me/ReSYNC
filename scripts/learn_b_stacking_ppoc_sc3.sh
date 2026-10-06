source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/conda/envs/maniskill3_env

python3 src/skill_refactor/main.py --seed 0 --approach rl_planning_states --env blocked_stacking \
    --exp_name skill_1112_sc3 \
    --lll_config config/lifelong_learning/blocked_stacking_sc3.yaml \
    --rl_algo PPOC \
    --control_mode pd_joint_delta_pos \
    --num_train_episodes_rl 100 \
    --num_envs 32 \
    --num_eval_envs 32 \
    --log_file logs/skill_1112_sc3.log
