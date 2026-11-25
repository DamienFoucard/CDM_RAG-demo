"""Exercise graph visibility against the page's actual JavaScript renderer."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_hidden_nodes_remove_incident_graph_edges_and_restore_cleanly():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for the dependency-free JavaScript graph test")
    subprocess.run(
        [node, str(ROOT / "tests/graph_visibility_harness.js"), str(ROOT / "web/graph.html")],
        check=True, capture_output=True, text=True,
    )
