# Intrinsic Reward Functions

This directory contains intrinsic reward functions for fast RL policy iteration without planner evaluation.

## Overview

By default, the `PlanningStatesVectorEnv` wrapper evaluates RL policies using the symbolic planner:
- RL policy acts for `skill_max_steps`
- Planner attempts to complete the task
- Reward is based on planner success/failure

For **faster iteration**, you can disable planner evaluation and use custom intrinsic rewards instead.

## Usage

### 1. Set Configuration

Either in your config file or via command line:

```bash
--planner_eval False \
--intrinsic_reward_path src/skill_refactor/approaches/intrinsic_reward/your_reward.py
```

Or in Python config:
```python
update_config({
    "planner_eval": False,
    "intrinsic_reward_path": "src/skill_refactor/approaches/intrinsic_reward/your_reward.py"
})
```

### 2. Create Reward Function

Your reward file must define an `intrinsic_rwd` function with this signature:

```python
import torch

def intrinsic_rwd(obs: torch.Tensor) -> torch.Tensor:
    """Compute intrinsic reward from observation.

    Args:
        obs: Full observation tensor (num_envs, obs_dim)
            NOT the clipped observation used by the RL policy

    Returns:
        Reward tensor (num_envs,) typically in range [-1, 1]
    """
    # Your reward logic here
    return reward
```

### 3. Examples

See provided examples:
- `template.py` - Detailed template with multiple example implementations
- `cluttered_drawer_example.py` - Task-specific example for cluttered drawer

## When to Use Intrinsic Rewards

**Use intrinsic rewards when:**
- You want fast policy iteration without planner overhead
- You have a clear proxy metric for success (e.g., distance to goal)
- You're debugging or prototyping new skills
- The planner is too slow for rapid experimentation

**Use planner evaluation when:**
- You want end-to-end task completion feedback
- The task has complex multi-step requirements
- You're doing final training/evaluation
- You want to ensure the policy enables symbolic planning

## Reward Design Tips

1. **Scale consistently**: Keep rewards in [-1, 1] range to match planner rewards
2. **Be differentiable**: Use smooth functions when possible (distance vs binary)
3. **Multi-objective**: Combine multiple factors with appropriate weights
4. **Debug**: Print reward values to verify they make sense
5. **Iterate**: Start simple, add complexity as needed

## Implementation Details

The intrinsic reward is computed when the RL episode truncates (after `skill_max_steps`):

```python
# In PlanningStatesVectorEnv.step()
if truncations.any():
    if CFG.planner_eval:
        # Run planner evaluation (slow)
        ...
    else:
        # Use intrinsic reward (fast)
        intrinsic_reward = self.intrinsic_reward_fn(original_obs)
        rw_tensor += intrinsic_reward
```

The reward function is dynamically loaded at wrapper initialization:
- Validates path is a Python file
- Imports module and extracts `intrinsic_rwd` function
- Calls function at each episode truncation
