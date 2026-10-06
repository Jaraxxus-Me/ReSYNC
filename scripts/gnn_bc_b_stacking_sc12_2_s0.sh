

# for seed in 0
# do
#     echo "Training BC-GNN WITH predicates for seed ${seed}..."
#     python3 src/skill_refactor/main.py \
#     --approach bc \
#     --bc_model gnn \
#     --bc_use_predicate_augmentation True \
#     --env blocked_stacking \
#     --use_wandb True \
#     --seed ${seed} \
#     --planner_dataset_path training_data/blocked_stacking/Planner_data/scenario12_2/seed_0 \
#     --log_file logs/training_bc_gnn_pred_sc12_2_seed${seed}.log \
#     --gnn_num_epochs 300 \
#     --num_eval_episodes 0
# done


# for seed in 0
# do
#     echo "Training BC-GNN WITHOUT predicates for seed ${seed}..."
#     python3 src/skill_refactor/main.py \
#     --approach bc \
#     --bc_model gnn \
#     --bc_use_predicate_augmentation False \
#     --env blocked_stacking \
#     --seed ${seed} \
#     --planner_dataset_path training_data/blocked_stacking/Planner_data/scenario12_2/seed_0 \
#     --log_file logs/bc_gnn_nopred_sc12_2_seed${seed}.log \
#     --gnn_num_epochs 300 \
#     --num_eval_episodes 0
# done
  
# pytest tests/blocked_stacking/approaches/test_bc_gnn.py::test_loading_bc_gnn_blocked_stacking_sc12_2 -v -s --log-cli-level=DEBUG

# Test WITHOUT predicates

# BC_USE_PREDICATE_AUGMENTATION=False pytest tests/blocked_stacking/approaches/test_bc_gnn.py::test_loading_bc_gnn_blocked_stacking_sc12_2 \
#     -v -s \
#     --log-cli-level=INFO

# Test WITH predicates
BC_USE_PREDICATE_AUGMENTATION=True pytest tests/blocked_stacking/approaches/test_bc_gnn.py::test_loading_bc_gnn_blocked_stacking_sc12_2 \
    -v -s \
    --log-cli-level=INFO