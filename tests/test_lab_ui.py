"""Optional Node harness for the actual lab selection and request handlers."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.local
def test_lab_ui_selection_regressions() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable")
    result = subprocess.run(
        [node, "--test", str(Path(__file__).with_suffix(".cjs"))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
