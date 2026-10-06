import os
from pathlib import Path

from gymnasium.envs.registration import register

PACKAGE_DIR = Path(__file__).parent.resolve()
PACKAGE_ASSET_DIR = PACKAGE_DIR / "assets"
EXT_LIB_DIR = PACKAGE_DIR / "ext"


def register_all_environments() -> None:
    # BlockedStacking2D env
    register(
        id="skill_ref/BlockedStacking2D-v0",
        entry_point="skill_refactor.benchmarks.blocked_stacking.blocked_stacking_env:BlockedStacking2DEnv",
        order_enforce=False,
        disable_env_checker=True,
    )

    # ClutteredDrawer env
    register(
        id="skill_ref/ClutteredDrawer-v1",
        entry_point="skill_refactor.benchmarks.cluttered_drawer.cluttered_drawer_env:ClutteredDrawerEnv",
        order_enforce=False,
        disable_env_checker=True,
    )

    # ClutteredRoom env
    register(
        id="skill_ref/ClutteredRoom-v1",
        entry_point="skill_refactor.benchmarks.cluttered_room.cluttered_room_env:ClutteredRoomEnv",
        order_enforce=False,
        disable_env_checker=True,
    )
    os.environ["MS_ASSET_DIR"] = str(PACKAGE_ASSET_DIR / "maniskill_assets")
    # ClutteredRoomHeld env
    register(
        id="skill_ref/ClutteredRoomForceHeld-v1",
        entry_point="skill_refactor.benchmarks.cluttered_room.cluttered_room_held_env:ClutteredRoomHeldEnv",
        order_enforce=False,
        disable_env_checker=True,
    )

    # IcyTransport2D env
    register(
        id="skill_ref/IcyTransport2D-v0",
        entry_point="skill_refactor.benchmarks.icy_transport.icy_transport_env:IcyTransport2DEnv",
        order_enforce=False,
        disable_env_checker=True,
    )
