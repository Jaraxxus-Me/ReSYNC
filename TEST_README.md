# Standalone Test Script Documentation

## Overview

`src/skill_refactor/test.py` is a standalone script for running lifelong learning tests similar to the unit tests in `tests/cluttered_drawer/approaches/test_lifelong_ref.py`, but with command-line configuration support.

## Features

- **Flexible seed configuration**: Test with single or multiple seeds
- **Scenario-based testing**: Support for single scenarios (sc1) and multi-scenario lifelong learning (sc12_2, sc123_3)
- **Configurable evaluation**: Customize number of episodes, max steps, and other parameters
- **Video recording**: Optional video recording for visual debugging
- **Comprehensive logging**: Detailed logging with configurable log levels
- **Results summary**: Automatic success rate calculation and averaging across seeds

## Prerequisites

Ensure you have:
1. Activated the virtual environment: `source .venv/bin/activate`
2. Installed the package: `pip install -e ".[develop]"`
3. Pre-trained policies and predicates (as required by your config YAML files)

## Usage

**Note**: All commands should be run from the project root directory.

### Basic Usage

```bash
# Test single scenario with one seed (using module syntax)
python -m skill_refactor.test \
    --lll_config config/lifelong_learning/cluttered_drawer_sc1.yaml \
    --scenario 1

# Alternative: direct execution
python src/skill_refactor/test.py \
    --lll_config config/lifelong_learning/cluttered_drawer_sc1.yaml \
    --scenario 1

# Test with multiple seeds
python -m skill_refactor.test --seeds 0,1,2 \
    --lll_config config/lifelong_learning/cluttered_drawer_sc1.yaml \
    --scenario 1

# Test multi-scenario lifelong learning
python -m skill_refactor.test \
    --lll_config config/lifelong_learning/cluttered_drawer_sc12_2.yaml \
    --scenario 12_2
```

### Advanced Usage

```bash
# Custom evaluation parameters
python -m skill_refactor.test \
    --lll_config config/lifelong_learning/cluttered_drawer_sc1.yaml \
    --scenario 1 \
    --num_eval_episodes 100 \
    --max_env_steps 500 \
    --num_envs 4

# Enable video recording (with the default evaluation seed only)
python -m skill_refactor.test \
    --lll_config config/lifelong_learning/cluttered_drawer_sc1.yaml \
    --scenario 1 \
    --save_video \
    --video_dir my_videos

# Custom logging
python -m skill_refactor.test \
    --lll_config config/lifelong_learning/cluttered_drawer_sc1.yaml \
    --scenario 1 \
    --log_file logs/my_test.log \
    --debug_log

# Test all seeds for scenario 12_2
python -m skill_refactor.test --seeds 0,1,2,3,4 \
    --lll_config config/lifelong_learning/cluttered_drawer_sc12_2.yaml \
    --scenario 12_2 \
    --num_eval_episodes 50
```

## Command-Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--lll_config` | Path to lifelong learning YAML config file | `config/lifelong_learning/cluttered_drawer_sc1.yaml` |
| `--scenario` | Scenario name | `1`, `12_2`, `123_3` |
| `--seed` or `--seeds` | Single seed or comma-separated list | `--seed <seed>` or `--seeds 0,1,2` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_eval_episodes` | 50 | Number of evaluation episodes per configuration |
| `--max_env_steps` | 300 | Maximum steps per episode |
| `--num_envs` | 1 | Number of parallel environments |
| `--save_video` | False | Save videos of episodes (only for first seed) |
| `--video_dir` | `videos` | Directory to save videos |
| `--log_file` | Auto-generated | Log file path (default: `logs/test_sc{scenario}_seed{seed}.log`) |
| `--debug_log` | False | Enable debug logging |
| `--control_mode` | `pd_joint_delta_pos` | Control mode |
| `--delta_finger_control` | False | Enable delta finger control |
| `--force_skip_pred_learning` | True | Skip predicate learning |

## Scenarios Explained

### Single Scenario (sc1)
Tests a single obstacle type (drawer) with 2 evaluation configurations:
- `1_b`: Drawer blocking stacking
- `1_g`: Drawer blocking grasp

### Two Scenarios (sc12_2)
Tests lifelong learning across 2 obstacle types (drawer + block) with 6 evaluation configurations:
- `1_b_2_b`, `1_b_2_g`, `1_g_2_b`, `1_g_2_g`: Combined obstacles
- `1_b`, `1_g`: Single obstacle baseline

### Three Scenarios (sc123_3)
Tests lifelong learning across 3 obstacle types (drawer + block + wall) with 12 evaluation configurations:
- `1_g_2_g_3_b`, `1_b_2_b_3_g`: Three obstacles
- Various two-obstacle combinations: `1_b_2_g`, `1_g_2_b`, etc.
- Single obstacle baselines: `1_b`, `1_g`

## Output

### Console Output
The script prints:
- Configuration details
- Per-episode success/failure
- Success rate for each evaluation configuration
- Final summary across all configurations
- Average success rates across seeds (when multiple seeds)

### Log Files
Detailed logs are saved to:
- Default: `logs/test_sc{scenario}_seed{seed}.log`
- Custom: Specified via `--log_file`

### Videos (Optional)
When `--save_video` is enabled (only with the default evaluation seed):
- Saved to: `{video_dir}/sc{scenario}_seed{seed}_{config_name}/`
- Format: MP4 at 30 FPS

## Example Workflows

### Quick Test (Single Seed)
```bash
python -m skill_refactor.test \
    --lll_config config/lifelong_learning/cluttered_drawer_sc1.yaml \
    --scenario 1 \
    --num_eval_episodes 10
```

### Comprehensive Evaluation (Multiple Seeds)
```bash
python -m skill_refactor.test --seeds 0,1,2,3,4 \
    --lll_config config/lifelong_learning/cluttered_drawer_sc12_2.yaml \
    --scenario 12_2 \
    --num_eval_episodes 50
```

### Debug Mode with Video
```bash
python -m skill_refactor.test \
    --lll_config config/lifelong_learning/cluttered_drawer_sc1.yaml \
    --scenario 1 \
    --num_eval_episodes 5 \
    --save_video \
    --debug_log
```

## Comparison with Unit Tests

| Feature | Unit Tests (`pytest`) | Standalone Test (`src/skill_refactor/test.py`) |
|---------|----------------------|---------------------------|
| Configuration | Hard-coded in test functions | Command-line arguments |
| Seeds | Parametrized in decorator | Command-line `--seed` or `--seeds` |
| Scenarios | Separate test functions | Single script with `--scenario` |
| Running | `pytest tests/...` | `python -m skill_refactor.test ...` |
| Flexibility | Low (requires code changes) | High (all via CLI) |
| Integration | CI/CD friendly | Research/debugging friendly |

## Troubleshooting

### Missing Config File
```
Error: [Errno 2] No such file or directory: 'config/...'
```
**Solution**: Ensure the config YAML file exists and the path is correct.

### Missing Pre-trained Policies
```
Error: Policy file not found: ...
```
**Solution**: Check the YAML config for `policy_path` and ensure the policy files exist.

### CUDA Out of Memory
```
Error: CUDA out of memory
```
**Solution**: Reduce `--num_envs` or set `device: "cpu"` in config.

### Planning Failures
```
Episode X failed during reset with TaskThenMotionPlanningFailure
```
**Solution**: This is expected behavior when the planner cannot find a valid plan. The episode is marked as failed and testing continues.

## Implementation Details

The script follows the same testing pattern as `tests/cluttered_drawer/approaches/test_lifelong_ref.py`:

1. **Configuration Setup**: Uses `reset_config()` to initialize settings
2. **TAMP System Creation**: Creates `ClutteredDrawerRLTAMPSystem` with default parameters
3. **Approach Initialization**: Creates `LifelongRefApproach` with the TAMP system
4. **Domain Knowledge Update**:
   - Single scenario: One-time update with `update_domain_knowledge()`
   - Multi-scenario: Sequential updates through all scenarios
5. **Evaluation Loop**: Tests multiple environment configurations
6. **Episode Execution**: Runs episodes with planning and skill execution
7. **Results Aggregation**: Calculates and reports success rates

## Notes

- Video recording is only enabled with the default evaluation seed to save disk space
- The script automatically creates log directories if they don't exist
- Planning failures during reset are caught and logged, not treated as crashes
- Success rate is calculated as: `num_successful_episodes / total_episodes`
- When testing multiple seeds, results are averaged across all seeds

## Future Enhancements

Possible improvements:
- Support for other environments (blocked_stacking, etc.)
- Parallel execution across seeds
- JSON/CSV output for automated analysis
- Integration with wandb or other logging platforms
- Custom evaluation configuration files
