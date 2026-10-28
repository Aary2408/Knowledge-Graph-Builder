from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentTracker:
    """Persists lightweight metadata about training and evaluation runs."""

    def __init__(self, root: str | Path = "checkpoints/experiments") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def record(self, name: str, payload: Dict[str, Any]) -> Path:
        target = self.root / f"{name}.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def list(self) -> List[Path]:
        return sorted(self.root.glob("*.json"))
