"""CPU checks for the optional demo; never launch or import IsaacLab."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def test_cli_import_does_not_load_simulator():
    code = "import sys; import resync_isaac.cli; assert not any(n.startswith(('isaaclab', 'omni', 'uwlab')) for n in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_default_four_blocks_and_pair_validation():
    from resync_isaac.cli import parser

    args = parser().parse_args(["--env", "flipped_stacking"])
    assert args.num_blocks == 4
    assert args.flipped is None
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/infer_isaac.py"),
            "--env",
            "flipped_stacking",
            "--flipped",
            "top",
            "top",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "two distinct roles" in result.stderr


def test_legacy_pose_architecture_is_unchanged():
    from skill_refactor.approaches.pred_learner.neural_models import setup_predicate_net
    from resync_isaac.neural_models import setup_predicate_net as demo_setup

    spec = {
        "type": "PoseMLP",
        "layer_size": 128,
        "initializer": "xavier",
        "input_dim": 7,
    }
    legacy = setup_predicate_net(spec)
    demo = demo_setup(dict(spec, type="IsaacPoseMLP"))
    assert legacy.encoder[0].in_features == 4
    assert demo.encoder[0].in_features == 3


@pytest.mark.parametrize(
    "domain,prefix,architecture",
    [
        ("flipped_stacking", "b_sc1p1_0", "IsaacPoseMLP"),
        ("flipped_stacking", "u_sc1p1_0", "UpAxisMLP"),
        ("corner_stacking", "b_sc1p2_0", "IsaacPoseMLP"),
    ],
)
def test_release_predicate_checkpoint(domain, prefix, architecture):
    from resync_isaac.neural_models import setup_predicate_net

    file = (
        ROOT / "inference_assets/isaac" / domain / "predicates" / f"{prefix}_model.pth"
    )
    if not file.exists():
        pytest.skip("Optional Isaac assets not downloaded")
    model = setup_predicate_net(
        {
            "type": architecture,
            "layer_size": 128,
            "initializer": "xavier",
            "input_dim": 7,
        }
    )
    model.load_state_dict(
        torch.load(file, map_location="cpu", weights_only=True), strict=True
    )
    state = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]]
    )
    output = model(state)
    assert output.shape == (1, 2, 1)
    assert torch.isfinite(output).all()
    if architecture == "UpAxisMLP":
        probabilities = torch.sigmoid(output)
        assert probabilities[0, 0, 0] > 0.99
        assert probabilities[0, 1, 0] < 0.01
