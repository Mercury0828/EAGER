"""Shared pytest fixtures and path setup for the EAGER test suite.

Test hygiene (guide §12): tests write only to pytest tmp dirs; config inputs
are read from the tracked configs/ tree or built as in-memory dicts.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for test helper modules


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def hardware_dir(repo_root) -> Path:
    return repo_root / "configs" / "hardware"


@pytest.fixture(scope="session")
def circuits_dir(repo_root) -> Path:
    return repo_root / "configs" / "circuits"
