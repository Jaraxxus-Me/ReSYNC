"""Opt-in GPU check, isolated so Kit shutdown cannot exit the pytest process."""

import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(
    os.environ.get("RESYNC_TEST_ISAAC") != "1",
    reason="Opt-in Isaac GPU integration check",
)
def test_all_six_reset_pairs():
    result = subprocess.run(
        [sys.executable, "-m", "resync_isaac.check_reset_pairs"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS all six pairs and three random-pair resets" in result.stdout
    print(result.stdout)
