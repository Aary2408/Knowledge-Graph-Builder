"""Vercel entrypoint for the existing KnowGraph AI Flask application."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from knowledge_graph_builder.app import create_app
from knowledge_graph_builder.graph import KnowledgeGraph


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
RUNTIME_ROOT = Path(os.getenv("KNOWGRAPH_RUNTIME_DIR", "/tmp/knowgraph"))
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)


def _runtime_copy(filename: str) -> Path:
    """Copy bundled seed data to the writable serverless filesystem once."""
    source = DATA_ROOT / filename
    target = RUNTIME_ROOT / filename
    if source.is_file() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return target


graph_seed = DATA_ROOT / "knowledge_graph.json"
graph = KnowledgeGraph.load_json(graph_seed) if graph_seed.is_file() else KnowledgeGraph("knowgraph")

app = create_app(
    graph,
    {
        "graph_path": str(_runtime_copy("knowledge_graph.json")),
        "documents_path": str(_runtime_copy("documents.json")),
        "uploads_path": str(RUNTIME_ROOT / "uploads"),
        "experiment_dir": str(RUNTIME_ROOT / "experiments"),
        "nlp_checkpoint_dir": str(RUNTIME_ROOT / "checkpoints" / "nlp"),
        "gnn_checkpoint_dir": str(RUNTIME_ROOT / "checkpoints" / "gnn"),
        "nli_checkpoint_dir": str(RUNTIME_ROOT / "checkpoints" / "verification"),
        "max_upload_bytes": int(os.getenv("KNOWGRAPH_MAX_UPLOAD_BYTES", str(16 * 1024 * 1024))),
    },
)
