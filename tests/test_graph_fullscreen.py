"""Exercise graph full-screen controls against the page's actual JavaScript."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_graph_fullscreen_entry_exit_and_browser_fallback():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for the dependency-free JavaScript graph test")
    subprocess.run(
        [node, str(ROOT / "tests/graph_fullscreen_harness.js"), str(ROOT / "web/graph.html")],
        check=True, capture_output=True, text=True,
    )
