"""Phase 1A acceptance: determinism across TWO SEPARATE PROCESS INVOCATIONS.

Runs scripts/run_episode.py twice in fresh subprocesses with identical
(config, seed, policy) and requires bit-identical stdout, including the
trajectory hash line. (Phase 1B extends this to a stochastic config.)
"""

import subprocess
import sys

import pytest


def run_episode_subprocess(repo_root, hardware, circuit, seed, extra=()):
    cmd = [sys.executable, str(repo_root / "scripts" / "run_episode.py"),
           "--hardware", hardware, "--circuit", circuit,
           "--seed", str(seed), "--policy", "jit", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=repo_root, timeout=300)
    assert proc.returncode == 0, f"run_episode failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


@pytest.mark.parametrize("circuit", ["configs/circuits/golden_micro_1.yaml",
                                     "configs/circuits/golden_micro_2.yaml"])
def test_two_process_invocations_identical_deterministic(repo_root, circuit):
    out1 = run_episode_subprocess(
        repo_root, "configs/hardware/golden_k2_det.yaml", circuit, seed=123)
    out2 = run_episode_subprocess(
        repo_root, "configs/hardware/golden_k2_det.yaml", circuit, seed=123)
    assert "trajectory_sha256=" in out1
    assert out1 == out2
