source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/conda/envs/maniskill3_env


for seed in 0
do
    echo "Running seed ${seed}..."
    python -m skill_refactor.test --seed ${seed} \
        --lll_config config/lifelong_learning/cluttered_drawer_sc12_2_seed${seed}.yaml \
        --max_env_steps 800 \
        --save_video \
        --log_file logs/skill_pred_0110_c_drawer_sc12_seed${seed}_eva.log \
        --control_mode pd_joint_delta_pos \
        --force_skip_pred_learning \
        --scenario "1,2"
done