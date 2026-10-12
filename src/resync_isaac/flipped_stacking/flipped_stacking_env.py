"""Environment factory for the Isaac Sim flipped-stacking benchmark."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers.event_manager import EventManager as _EventManager
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f140.rl_state_cfg import (
    Ur5eRobotiq2f140RelCartesianOSCEvalCfg,
)

from resync_isaac.corner_stacking.corner_stacking_env import (
    ACTION_SCALE,
    ScriptedBringToGoal,
)
from resync_isaac.flipped_stacking.task_cfg import (
    FLIPPED_STACKING_TASK_ID,
    POLICY_DT,
    FlippedStackingCfg,
)
from skill_refactor.settings import CFG

# Keep class-based reset terms reliable in headless Isaac Lab startup.
if not getattr(_EventManager, "_skill_refactor_apply_patched", False):
    _original_event_manager_apply = _EventManager.apply

    def _event_manager_apply_with_resolution(self, *args, **kwargs):
        if not getattr(self, "_is_scene_entities_resolved", True):
            self._resolve_terms_callback(None)
        return _original_event_manager_apply(self, *args, **kwargs)

    _EventManager.apply = _event_manager_apply_with_resolution
    _EventManager._skill_refactor_apply_patched = True


def make_flipped_stacking_env(
    num_envs: int | None = None,
    max_steps: int | None = None,
    device: str | None = None,
    env_cfg: ManagerBasedRLEnvCfg | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Construct the two-cube flipped-stacking environment."""
    if num_envs is None:
        num_envs = int(CFG.fstack_num_envs)
    if max_steps is None:
        max_steps = int(CFG.fstack_max_steps)
    if env_cfg is None:
        env_cfg = FlippedStackingCfg()
        # Pin the exact robot and action term used by the ReSYNC checkpoint.
        eval_cfg = Ur5eRobotiq2f140RelCartesianOSCEvalCfg()
        env_cfg.scene.robot = eval_cfg.scene.robot.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        env_cfg.actions = eval_cfg.actions
    # Configclass fields are materialized when the module is imported, while
    # tests and experiment launchers intentionally switch this setting at
    # runtime.  Refresh the role on every factory call.
    env_cfg.events.reset_task.params["flipped_cube"] = str(CFG.fstack_flipped_cube)
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.episode_length_s = float(max_steps) * POLICY_DT
    if device is not None:
        env_cfg.sim.device = device
    from resync_isaac.runtime import configure_scene

    configure_scene(env_cfg)
    kwargs: dict[str, Any] = {"cfg": env_cfg}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gym.make(FLIPPED_STACKING_TASK_ID, **kwargs)


__all__ = [
    "ACTION_SCALE",
    "FLIPPED_STACKING_TASK_ID",
    "ScriptedBringToGoal",
    "make_flipped_stacking_env",
]
