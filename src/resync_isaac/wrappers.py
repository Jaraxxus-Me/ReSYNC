"""Isaac policy observation projection (loaded only in the demo runtime)."""

from typing import Any, Callable
import torch

PolicyObsAdapter = Callable[[torch.Tensor], torch.Tensor]


def prefix_slice_adapter(num_cols: int) -> PolicyObsAdapter:
    """Adapter that keeps the first ``num_cols`` columns of the policy obs.

    Use when the canonical PolicyCfg appends new terms after the training
    layout (no reordering). The adapter is a pure slice — no allocations,
    no copy of the original data. See ``SkillVecEnvWrapper`` for the
    invariants that must hold for the prefix to match byte-for-byte.
    """
    num_cols = int(num_cols)

    def _adapt(obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] < num_cols:
            raise ValueError(
                f"prefix_slice_adapter expected obs width >= {num_cols}, "
                f"got {obs.shape[-1]}"
            )
        return obs[..., :num_cols]

    return _adapt


_SKILL_VEC_ENV_WRAPPER_CLS: Any = None


def _build_skill_vec_env_wrapper_cls() -> Any:
    """Lazily construct ``SkillVecEnvWrapper`` on first use.

    Importing ``isaaclab_rl`` transitively pulls in ``isaaclab.envs`` which
    requires ``AppLauncher`` to have been started first. Deferring the
    import until the factory is invoked keeps ``wrappers.py`` usable in
    non-Isaac contexts.
    """
    global _SKILL_VEC_ENV_WRAPPER_CLS
    if _SKILL_VEC_ENV_WRAPPER_CLS is not None:
        return _SKILL_VEC_ENV_WRAPPER_CLS

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore
    from tensordict import TensorDict  # type: ignore

    class _SkillVecEnvWrapper(RslRlVecEnvWrapper):  # type: ignore[misc]
        """RslRl wrapper that projects canonical policy obs to a skill's layout.

        The underlying env emits a superset ``PolicyCfg``. On each
        ``get_observations``/``reset``/``step``, the bound ``obs_adapter``
        slices or reorders the policy tensor into the exact layout the
        skill's policy was trained on. Critic group is untouched (the
        canonical ``CriticCfg`` is kept equal to the training cfg, so no
        critic adapter is needed at inference time).

        Notes:
            * Pass a different adapter per skill — either keep one wrapper
              per skill (cheap: raw env is shared, only reset/step dispatch
              differ) or swap ``.obs_adapter`` on a single wrapper between
              skill invocations within an episode.
            * For training a skill's policy end-to-end, train on a narrow
              task cfg whose ``PolicyCfg`` already matches the skill's
              layout — the adapter is unnecessary there and the wrapper
              degrades to a plain ``RslRlVecEnvWrapper``.
        """

        def __init__(
            self,
            env,
            obs_adapter: PolicyObsAdapter | None = None,
            clip_actions: float | None = None,
        ) -> None:
            super().__init__(env, clip_actions=clip_actions)
            self.obs_adapter = obs_adapter

        def _project(self, obs_td):
            if self.obs_adapter is None:
                return obs_td
            out = dict(obs_td)
            policy = out.get("policy")
            if policy is not None:
                out["policy"] = self.obs_adapter(policy)
            return TensorDict(out, batch_size=[self.num_envs])

        def get_observations(self):  # type: ignore[override]
            return self._project(super().get_observations())

        def reset(self):  # type: ignore[override]
            obs_td, info = super().reset()
            return self._project(obs_td), info

        def step(self, actions):  # type: ignore[override]
            obs_td, rew, dones, extras = super().step(actions)
            return self._project(obs_td), rew, dones, extras

    _SKILL_VEC_ENV_WRAPPER_CLS = _SkillVecEnvWrapper
    return _SKILL_VEC_ENV_WRAPPER_CLS


def SkillVecEnvWrapper(
    env,
    obs_adapter: PolicyObsAdapter | None = None,
    clip_actions: float | None = None,
):
    """Factory for the skill-specific rsl_rl wrapper; see module docstring.

    Used *in place of* ``RslRlVecEnvWrapper`` when running a pretrained
    skill on a canonical/superset env. Returns an instance of a subclass
    constructed lazily on first call.
    """
    cls = _build_skill_vec_env_wrapper_cls()
    return cls(env, obs_adapter=obs_adapter, clip_actions=clip_actions)
