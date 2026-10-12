"""Isolated GPU integration check for all flipped-pair selections."""

import os
from itertools import combinations


def main() -> None:
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, fast_shutdown=True).app
    system = None
    try:
        from resync_isaac.settings import configure

        configure(
            device="cuda:0",
            seed=0,
            num_of_blocks=4,
            fstack_num_envs=1,
            fstack_flipped_cube="random_pair",
        )
        from resync_isaac.flipped_stacking.flipped_stacking import (
            FlippedStackingRLTAMPSystem,
        )
        from resync_isaac.flipped_runner import _is_upright

        system = FlippedStackingRLTAMPSystem.create_default(seed=0)
        roles = ("top", "mid1", "mid2", "bottom")
        event = system.env.unwrapped.event_manager.get_term_cfg("reset_task")
        for pair in combinations(roles, 2):
            event.params["flipped_cube"] = ",".join(pair)
            system.env.reset()
            actual = {role for role in roles if not _is_upright(system, role)}
            assert actual == set(pair), (pair, actual)
            print(f"PASS reset {pair}: {sorted(actual)}", flush=True)
        event.params["flipped_cube"] = "random_pair"
        for _ in range(3):
            system.env.reset()
            assert sum(not _is_upright(system, role) for role in roles) == 2
    except BaseException:
        import traceback

        traceback.print_exc()
        import sys

        sys.stderr.flush()
        os._exit(1)
    else:
        print("PASS all six pairs and three random-pair resets", flush=True)
        if system is not None:
            system.env.close()
        app.close()


if __name__ == "__main__":
    main()
