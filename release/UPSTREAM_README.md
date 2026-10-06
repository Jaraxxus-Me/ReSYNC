# Lifelong Skill Refactorization

An agent that **endlessly** learn new **skills** (neural), planning **predicates** (neural), and **operators** (symbolic) by interacting with a physical environment.
This is a mixture of Hierachical Reinforcement Learning (Option Learning, Terminal and Initiation Functions), and Hierachical Planning (Operator, Predicate Learning).

Our key desideratas of the system:
- A skill can be used differently from how it was learned during training (refactorization -> generalization).
- Skills learned from different environments can be zero-shot composed to solve unseen novelty compositions. (refactorization -> compositionality).

## :octocat: Contributing

### :ballot_box_with_check: Requirements
1. Python >=3.10, <3.13
2. Tested on Ubuntu 22.04 with RTX 3090 GPU, right now it has to be at least a linux machine due to Maniskill requirements.

### :wrench: Installation
We strongly recommend [uv](https://docs.astral.sh/uv/getting-started/installation/). The steps below assume that you have `uv` installed. If you do not, just remove `uv` from the commands and the installation should still work.
```
# Install this repo
git clone https://github.com/Jaraxxus-Me/skill_refactor.git
cd skill_refactor
git submodule update --init
```
```
# Create venv
uv venv --python=3.11
source .venv/bin/activate
# Install skill_ref
uv pip install -e .[develop]
# Third-party dependencies
uv pip install -e third-party/prpl-mono/prpl-utils
uv pip install -e third-party/prpl-mono/prbench
uv pip install -e third-party/prpl-mono/relational-structs
uv pip install -e third-party/prpl-mono/toms-geoms-2d
```

### :microscope: Check Installation
Run `./run_ci_checks.sh`. It should complete with all green successes.
Additionally, see `videos` for a simple dynamic2d execution video with given planner :)

To verify advanced capability:

0. Checkout to `stable` branch (10/25/2025 MileStone)
    ```
    git checkout stable
    # update dependencies
    git submodule update
    ```

1. Download pre-trained predicates, skills, and data (takes ~1GB disk space)
    ```
    python3 scripts/download_2d.py
    ```
    You should expect five folders:
    `skill_*_pred_nets_seed*`: The discovered neural predicates and learned operators for all scenarios and all seeds.
    `trained_policies`: The learned skill policies in for all scenarios and all seeds.
    `training_data`: The data needed to learn the RL skill policy and the data needed to do predicate discovery and planner refactorization.
    `videos`: Some videos from evaluation.
    `logs`: Raw outputs for training and evaluation of the policies and predicates discussed above.

2. Use a unit test that was skiped:
    ```shell
    # Comment the pytest.mark.skip line.
    # Mix RL skill with Planning Abstraction for OOD Decision Making :)
    # See L83-93, L219-284, L412-575 about evaluation settings.
    # SC1
    pytest -s -v tests/blocked_stacking/approaches/test_lifelong_ref.py::test_loading_learned_skill_predicate_blocked_stacking_sc1_or_sc2_or_sc3
    # SC12_2
    pytest -s -v tests/blocked_stacking/approaches/test_lifelong_ref.py::test_loading_learned_skill_predicate_blocked_stacking_sc12_2
    # SC123_3
    pytest -s -v tests/blocked_stacking/approaches/test_lifelong_ref.py::test_loading_learned_skill_predicate_blocked_stacking_sc123_3
    ```
    You should see (most of) the episodes are successful, check the video at `videos`.

### :mag: General Guidelines
* All checks must pass before code is merged (see `./run_ci_checks.sh`)
* All code goes through the pull request review process:
    1. create a custom branch `others`.
    2. Do whatever you want, but need to pass all CI checks.
    3. Push to origin `git push origin others`.
    4. Request a review -> Merge into `main`.
* Branches:
    - `main`: Latest development code that has passed all ci checks. Any new branch should start from this branch and be merged into this branch.
    - `stable`: Older code that is verified and can reproduce some results for understanding. We will update it periodically.
    - `others`: Experimental branches that have not being merged into main.
* Code understanding (for dynamic2d, sc1):
    - `tests/blocked_stacking/datasets/test_collect.py::test_blocked_stacking_rl_data_collection_sc1_or_2`: collecting RL data with current planner (by augmenting provided tasks).
    - `tests/blocked_stacking/benchmarks/test_wrapper.py::test_rl_planning_wrapper_blocked_stacking_sc1_or_2_or_3`: training RL policy with current planner (by planning-conditioned failure recovery).
    - `tests/blocked_stacking/datasets/test_collect.py::test_blocked_stacking_planner_data_collection_sc1_or_2_or_3`: collecting planner data (by compositional dreaming).
    -  `tests/blocked_stacking/approaches/test_pred_learning_topdown.py::test_fixed_predicate_invention_blocked_stacking_sc1_or_2_or_3`: discovering new predicates and refactoring planning operators (by 3-dim effect enumeration IVNTR).
    - `tests/blocked_stacking/approaches/test_lifelong_ref.py::test_loading_learned_skill_predicate_blocked_stacking_sc1_or_sc2_or_sc3`: evaluating the new planner with new skills and predicates.
    - For sc12_2, sc123_3, you will also find corresponding unit test files.
    - To run the entire training in on command line, checkout:
        `scripts/learn_b_stacking_ppoc_sc1_s*.sh, learn_b_stacking_ppoc_sc12_2_s*.sh, learn_b_stacking_ppoc_sc123_3_s*.sh`.

* Simulator understanding (for ClutteredDrawer):
    - Make sure you are on a Linux machine with GPU (Mac OS should be good with CPU simulation, but not tested).
    - Understand the following Unit Tests in order:
        ```shell
        # Main Branch
        git checkout main

        # Basic definition of maniskill environments and batched simulation
        pytest -s -v tests/cluttered_drawer/benchmarks/test_cluttereddrawer_env.py

        # Spot motion skills: 6-axis arm and tidybot base, TidySpot :)
        pytest -s -v tests/cluttered_drawer/utils/test_spot_motion.py

        # Run a batched task-then-motion-planner in Maniskills
        pytest -s -v tests/cluttered_drawer/approaches/test_pure_tamp.py::test_clutted_drawer_base_tamp
        ```
    - Implementing a new env requires two files:
        - `src/skill_refactor/benchmarks/cluttered_drawer/cluttered_drawer_env.py`: An environment that a robot can interact with, based on `BaseEnv` from Maniskill.
        - `src/skill_refactor/benchmarks/cluttered_drawer/cluttered_drawer.py`: Types, Predicates, Skills, and Operator definition for the environment.

* Cluttered-Room (Maniskill-HAB) asset and task setup:

    We have implemented (borrowed) the API for evaluting with the tasks in Maniskill-HAB as the starting environment for this project.

    Download the ReplicaCAD dataset necessary for low-level manipulation, which can be downloaded with ManiSkill's download utils. This may take some time:
    ```bash
    # Note that this path is forced
    export MS_ASSET_DIR="[path_to_this_repo]/src/skill_refactor/assets/maniskill_assets"
    for dataset in ycb ReplicaCAD ReplicaCADRearrange; do python -m mani_skill.utils.download_asset "$dataset"; done
    ```

    Download the Maniskill-HAB dataset:
    ```bash
    huggingface-cli login   # in case not already authenticated

    # Dataset (see HuggingFace documentation for faster download options depending on your system)
    export MS_ASSET_DIR="[path_to_this_repo]/src/skill_refactor/assets/maniskill_assets" # change to your preferred path (if changed, ideally add to .bashrc)
    export MSHAB_DATASET_DIR="$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange-dataset"
    huggingface-cli download --repo-type dataset arth-shukla/MS-HAB-TidyHouse --local-dir "$MSHAB_DATASET_DIR/tidy_house"
    huggingface-cli download --repo-type dataset arth-shukla/MS-HAB-PrepareGroceries --local-dir "$MSHAB_DATASET_DIR/prepare_groceries"
    huggingface-cli download --repo-type dataset arth-shukla/MS-HAB-SetTable --local-dir "$MSHAB_DATASET_DIR/set_table"
    ```


### :oncoming_automobile: Roadmap
Before submission, here are some low-level TODOs:

**Infra**
- [x] Finalize 2D environment results, data, and all of the models. (So that baselines can be tested)
- [x] Build up 3D envoronment infra: Maniskills + TidySpot

**Baselines**
- [ ] Implement baseline: [SOL]() (pure neural), with access to all of our skills, trained on our tasks.
- [ ] Implement baseline: SOL (pure neural), with access to all of our skills + some new skills, trained on our tasks.
- [ ] Implement baseline: [DSG]() (neural skill + terminal function based high-level planning), with access to all of our skills, trained on our tasks, with the same number of total max env steps.
- [ ] Implement baseline: DSG (neural skill + terminal function based high-level planning), with access to all of our skills + allowing learning new skills, trained on our tasks, with the same number of total max env steps.
- [ ] Implement baseline: [GNN/TF BC]() (pure neural), with access to all of our skills, trained on our tasks-trajectories.
  - DSG note: set `CFG.dsg_mode=True` and point `planner_learning_cfg_settings.predicate_config` to `config/predicates/blocked_stacking_dsg_*.yaml`; this reuses the standard pyperplan stack with terminal predicates only.

**Env Dev**
- [ ] Build and finalize ClutteredDrawer environment (pure arm skills).
    - [x] Pull
    - [ ] Wiggle
    - [ ] Fiddle
- [ ] Build and finalize ClutteredRoom environment.
    - [ ] Push
    - [ ] Wipe/Tool use
    - [ ] ??
