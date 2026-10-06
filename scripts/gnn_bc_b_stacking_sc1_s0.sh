# for seed in 0
# do
#     echo "Training BC-GNN WITH predicates for seed ${seed}..."
#     python3 src/skill_refactor/main.py \
#     --approach bc \
#     --bc_model gnn \
#     --bc_use_predicate_augmentation True \
#     --use_wandb True \
#     --env blocked_stacking \
#     --seed ${seed} \
#     --planner_dataset_path training_data/blocked_stacking/Planner_data/scenario1/seed_0 \
#     --log_file logs/bc_gnn_pred_sc1_seed_0_of_data.log \
#     --gnn_num_epochs 300 \
#     --num_eval_episodes 0
# done

BC_USE_PREDICATE_AUGMENTATION=True USE_WANDB=1 pytest tests/blocked_stacking/approaches/test_bc_gnn.py::test_loading_bc_gnn_blocked_stacking_sc1 \
    -v -s \
    --log-cli-level=INFO 