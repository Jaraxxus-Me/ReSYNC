"""Single-scene runtime settings, independent of reproduction settings."""

from typing import Any


def configure_scene(cfg: Any) -> None:
    # Training defaults reserve gigabytes for thousands of parallel scenes.
    # One demo scene needs only a small fraction of those contact buffers.
    from skill_refactor.settings import CFG

    cfg.seed = CFG.seed
    cfg.sim.device = CFG.device
    physx = cfg.sim.physx
    physx.gpu_collision_stack_size = 2**26
    physx.gpu_heap_capacity = 2**26
    physx.gpu_temp_buffer_capacity = 2**24
    physx.gpu_max_rigid_contact_count = 2**18
    physx.gpu_max_rigid_patch_count = 2**16
    physx.gpu_found_lost_pairs_capacity = 2**18
    physx.gpu_found_lost_aggregate_pairs_capacity = 2**18
    physx.gpu_total_aggregate_pairs_capacity = 2**18
    cfg.viewer.resolution = (640, 480)
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.render.rendering_mode = "performance"

    from resync_isaac.paths import ASSET_ROOT

    mirrors = {
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/": "nvidia",
        "https://huggingface.co/datasets/bowenli1024/ReSYNC_IsaacLab/resolve/main/": "custom",
        "https://huggingface.co/datasets/bowenli1024/physcoder_usd/resolve/main/": "objects",
        "https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/": "uwlab",
    }
    for name, asset in vars(cfg.scene).items():
        spawn = getattr(asset, "spawn", None)
        path = getattr(spawn, "usd_path", None)
        if not path:
            continue
        assert spawn is not None
        for prefix, folder in mirrors.items():
            if path.startswith(prefix):
                local = ASSET_ROOT / "simulation" / folder / path.removeprefix(prefix)
                if not local.is_file():
                    raise FileNotFoundError(f"Missing demo scene asset {name}: {local}")
                spawn.usd_path = str(local)
                break

    sky = (
        ASSET_ROOT
        / "simulation/nvidia/Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
    )
    if not sky.is_file():
        raise FileNotFoundError(sky)
    cfg.scene.sky_light.spawn.texture_file = str(sky)

    # Presentation lighting only; no changes to the task's physical scene.
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim import DistantLightCfg

    cfg.scene.demo_key_light = AssetBaseCfg(
        prim_path="/World/DemoKeyLight",
        spawn=DistantLightCfg(intensity=2500.0, color=(1.0, 0.95, 0.9)),
    )
    cfg.viewer.eye = (-1.45, 0.8, 0.85)
    cfg.viewer.lookat = (-0.4, 0.0, 0.12)
