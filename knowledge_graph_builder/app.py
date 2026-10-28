from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, Optional

try:
    from flask import Flask, jsonify, request
except ImportError:  # pragma: no cover - optional dependency
    Flask = None

from .document_ingest import extract_document_text
from .experiments import ExperimentTracker
from .gnn import GraphNeuralNetworkTrainer
from .graph import KnowledgeGraph
from .nlp import EntityRecognizer, NLPTrainingPipeline, RelationExtractor
from .rag import GraphRAGEngine
from .verifier import ClaimNLITrainer, ClaimVerifier, TransformerNLIClassifier


def _text_chunks(text: str, max_characters: int = 1200) -> list[tuple[int, str]]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_characters, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + max_characters // 2, end)
            newline = text.rfind("\n", start + max_characters // 2, end)
            boundary = max(boundary, newline)
            if boundary > start:
                end = boundary
        chunk = text[start:end]
        if chunk.strip():
            chunks.append((start, chunk))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    return chunks


def create_app(graph: Optional[KnowledgeGraph] = None, model_config: Optional[Dict[str, Any]] = None) -> Any:
    if Flask is None:
        raise RuntimeError("Flask is required to create the web application. Install it with `pip install Flask`.")
    app = Flask(__name__, static_folder=str(Path(__file__).with_name("static")))
    settings = dict(model_config or {})
    max_upload_bytes = int(settings.get("max_upload_bytes", 16 * 1024 * 1024))
    if max_upload_bytes < 1:
        raise ValueError("max_upload_bytes must be positive.")
    app.config["MAX_CONTENT_LENGTH"] = max_upload_bytes + 128 * 1024
    graph_path = Path(settings["graph_path"]) if settings.get("graph_path") else None
    if graph is None and graph_path and graph_path.is_file():
        state_graph = KnowledgeGraph.load_json(graph_path)
    else:
        state_graph = graph or KnowledgeGraph("default")
    nlp_config = dict(settings.get("nlp", {}))
    if "dataset_path" not in nlp_config and settings.get("dataset_path"):
        nlp_config["dataset_path"] = settings["dataset_path"]
    nlp_config.setdefault("checkpoint_dir", settings.get("nlp_checkpoint_dir", "checkpoints/nlp"))
    nlp_config["checkpoint_dir"] = str(Path(nlp_config["checkpoint_dir"]).expanduser().resolve())
    nlp_pipeline = NLPTrainingPipeline(nlp_config)
    gnn_config = dict(settings.get("gnn", {}))
    gnn_config.setdefault("checkpoint_dir", settings.get("gnn_checkpoint_dir", "checkpoints/gnn"))
    nli_config = dict(settings.get("nli", {}))
    nli_config.setdefault("checkpoint_dir", settings.get("nli_checkpoint_dir", "checkpoints/verification"))
    if settings.get("nli_checkpoint"):
        nli_config["nli_checkpoint"] = settings["nli_checkpoint"]
    verifier = ClaimVerifier(state_graph, nli_config)
    rag = GraphRAGEngine(state_graph)
    tracker = ExperimentTracker(settings.get("experiment_dir", "checkpoints/experiments"))
    documents_path = Path(settings.get("documents_path", "data/documents.json"))
    uploads_path = Path(settings.get("uploads_path", "data/uploads"))
    if documents_path.is_file():
        stored_documents = json.loads(documents_path.read_text(encoding="utf-8"))
        documents: Dict[str, Dict[str, Any]] = {
            item["document_id"]: item for item in stored_documents
        }
    else:
        documents = {}
    executor = ThreadPoolExecutor(max_workers=int(settings.get("training_workers", 2)))
    jobs: Dict[str, Dict[str, Any]] = {}
    lock = threading.Lock()

    def persist(job_id: str) -> None:
        job = jobs[job_id]
        tracker.record(job_id, {key: value for key, value in job.items() if key not in {"future", "retry_run"}})

    def start_job(task: str, run: Any, retry_of: Optional[str] = None) -> str:
        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "task": task,
            "status": "queued",
            "logs": ["Training job queued."],
            "result": None,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "retry_of": retry_of,
            "retry_run": run,
        }
        with lock:
            jobs[job_id] = job
            persist(job_id)

        def execute() -> None:
            with lock:
                job["status"] = "running"
                job["logs"].append("Training started.")
                persist(job_id)

            def report(message: str) -> None:
                with lock:
                    job["logs"].append(message)
                    persist(job_id)

            try:
                result = run(report)
                with lock:
                    job["result"] = result
                    job["status"] = "completed" if result.get("status") == "completed" else result.get("status", "completed")
                    job["logs"].append("Training finished.")
                    persist(job_id)
            except Exception as error:
                with lock:
                    job["status"] = "failed"
                    job["error"] = str(error)
                    job["logs"].append(f"Training failed: {error}")
                    persist(job_id)

        with lock:
            job["future"] = executor.submit(execute)
        return job_id

    def checkpoint_path(raw_path: str, root: str | Path) -> Path:
        allowed_root = Path(root).resolve()
        candidate = Path(raw_path).resolve()
        if candidate != allowed_root and allowed_root not in candidate.parents:
            raise ValueError("Checkpoint path must be inside the configured checkpoint directory.")
        if not candidate.exists():
            raise FileNotFoundError(f"Checkpoint not found: {candidate}")
        return candidate

    def persist_documents() -> None:
        documents_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = documents_path.with_suffix(documents_path.suffix + ".tmp")
        temporary.write_text(json.dumps(list(documents.values()), indent=2), encoding="utf-8")
        temporary.replace(documents_path)

    @app.errorhandler(413)
    def upload_too_large(_error: Any) -> Any:
        return jsonify({"error": f"Upload exceeds the configured {max_upload_bytes}-byte limit."}), 413

    def persist_graph() -> None:
        if graph_path:
            state_graph.save_json(graph_path)

    def list_checkpoints(root: str | Path, suffix: str) -> list[str]:
        path = Path(root)
        if not path.exists():
            return []
        if suffix == ".pt":
            return sorted(str(item.resolve()) for item in path.glob("*.pt") if item.is_file())
        return sorted(
            str(item.resolve())
            for item in path.iterdir()
            if item.is_dir() and (item / "metadata.json").is_file()
        )

    def nlp_checkpoint_catalog() -> list[Dict[str, Any]]:
        catalog = []
        root = nlp_pipeline.checkpoint_dir
        if not root.exists():
            return catalog
        for path in sorted(root.iterdir()):
            if not path.is_dir() or not (path / "metadata.json").is_file():
                continue
            item: Dict[str, Any] = {"path": str(path.resolve()), "compatible": False}
            try:
                metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
                task = metadata.get("task")
                architecture = metadata.get("architecture", "")
                item.update({
                    "task": task,
                    "architecture": architecture,
                    "synthetic_data": bool(metadata.get("synthetic_data", False)),
                    "trained_at": metadata.get("trained_at"),
                })
                if task == "ner" and architecture == "local-context-feature-classifier":
                    valid = (path / "weights.pt").is_file() and isinstance(metadata.get("feature_to_id"), dict)
                elif task == "ner":
                    valid = (path / "config.json").is_file() and any(
                        (path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
                    )
                elif task == "relation":
                    valid = (path / "weights.pt").is_file() and isinstance(metadata.get("vocabulary"), dict)
                else:
                    valid = False
                item["compatible"] = bool(valid)
                if not valid:
                    item["reason"] = "Checkpoint files or task metadata are incomplete or unsupported."
            except (OSError, json.JSONDecodeError) as error:
                item["reason"] = f"Could not read checkpoint metadata: {error}"
            catalog.append(item)
        return catalog

    def public_document(document: Dict[str, Any]) -> Dict[str, Any]:
        public = {key: value for key, value in document.items() if key != "text"}
        public["preview"] = document.get("text", "")[:240]
        public["text_length"] = len(document.get("text", ""))
        return public

    @app.get("/")
    def index() -> Any:
        return app.send_static_file("index.html")

    @app.get("/health")
    def health() -> Any:
        return jsonify({
            "status": "ok",
            "graph": state_graph.name,
            "entities": len(state_graph.entities),
            "relationships": len(state_graph.relations),
            "documents": len(documents),
        })

    @app.get("/overview")
    def overview() -> Any:
        experiments = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in tracker.list()
        ]
        experiments.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return jsonify({
            "graph": {
                "name": state_graph.name,
                "entities": len(state_graph.entities),
                "relationships": len(state_graph.relations),
            },
            "documents": len(documents),
            "recent_experiments": experiments[:5],
            "activity": [
                {"type": "document", "document_id": item["document_id"], "status": item["status"], "updated_at": item["updated_at"]}
                for item in sorted(documents.values(), key=lambda row: row["updated_at"], reverse=True)[:5]
            ],
        })

    @app.get("/datasets")
    def list_datasets() -> Any:
        root = Path(settings.get("dataset_dir", "data"))
        items = []
        if root.exists():
            for path in sorted(root.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    items.append({"name": path.name, "path": str(path.resolve()), "status": "invalid", "error": str(error)})
                    continue
                rows = payload.get("examples", []) if isinstance(payload, dict) else []
                if not isinstance(payload, dict) or not isinstance(rows, list) or not rows:
                    continue
                items.append({
                    "name": path.name,
                    "path": str(path.resolve()),
                    "status": "available",
                    "dataset": payload.get("dataset") if isinstance(payload, dict) else None,
                    "examples": len(rows) if isinstance(rows, list) else None,
                    "synthetic": bool(payload.get("synthetic", False)) if isinstance(payload, dict) else False,
                })
        return jsonify({"datasets": items})

    @app.get("/models")
    def list_models() -> Any:
        catalog = nlp_checkpoint_catalog()
        compatible = [item for item in catalog if item["compatible"]]
        return jsonify({
            "nlp": [item["path"] for item in compatible],
            "nlp_by_task": {
                task: [item["path"] for item in compatible if item.get("task") == task]
                for task in ("ner", "relation")
            },
            "nlp_checkpoints": catalog,
            "gnn": list_checkpoints(gnn_config.get("checkpoint_dir", "checkpoints/gnn"), ".pt"),
            "nli": list_checkpoints(nli_config.get("checkpoint_dir", "checkpoints/verification"), ""),
        })

    @app.post("/models/load")
    def load_model() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        if (
            not isinstance(payload, dict)
            or payload.get("task") != "nli"
            or not isinstance(payload.get("checkpoint"), str)
            or not payload["checkpoint"]
        ):
            return jsonify({"error": "task `nli` and a checkpoint path are required."}), 400
        try:
            checkpoint = checkpoint_path(payload["checkpoint"], nli_config.get("checkpoint_dir", "checkpoints/verification"))
            classifier = TransformerNLIClassifier(checkpoint)
            verifier.nli_classifier = classifier
            verifier.config["nli_checkpoint"] = str(checkpoint)
            return jsonify({"status": "loaded", "task": "nli", "checkpoint": str(checkpoint)})
        except (OSError, ValueError, RuntimeError) as error:
            return jsonify({"error": str(error)}), 400

    @app.get("/documents")
    def list_documents() -> Any:
        items = sorted(documents.values(), key=lambda row: row["updated_at"], reverse=True)
        return jsonify({"documents": [public_document(item) for item in items]})

    @app.post("/documents")
    def create_document() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "A non-empty document text is required."}), 400
        document_id = payload.get("document_id") or uuid.uuid4().hex
        if not isinstance(document_id, str) or not document_id.strip():
            return jsonify({"error": "document_id must be a non-empty string."}), 400
        if document_id in documents:
            return jsonify({"error": f"Document {document_id!r} already exists."}), 409
        now = datetime.now(timezone.utc).isoformat()
        documents[document_id] = {
            "document_id": document_id,
            "text": text,
            "status": "stored",
            "entities": [],
            "candidate_relations": [],
            "created_at": now,
            "updated_at": now,
        }
        persist_documents()
        return jsonify({"document": public_document(documents[document_id])}), 201

    @app.post("/documents/upload")
    def upload_document() -> Any:
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify({"error": "Choose a document file to upload."}), 400
        filename = Path(uploaded.filename.replace("\\", "/")).name
        content = uploaded.read(max_upload_bytes + 1)
        if len(content) > max_upload_bytes:
            return jsonify({"error": f"Upload exceeds the configured {max_upload_bytes}-byte limit."}), 413
        try:
            text = extract_document_text(filename, content)
        except (RuntimeError, ValueError) as error:
            return jsonify({"error": str(error)}), 400
        if not text.strip():
            return jsonify({"error": "No text could be extracted from this file. Scanned PDFs require OCR before upload."}), 400
        document_id = request.form.get("document_id", "").strip() or uuid.uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", document_id) or document_id in {".", ".."}:
            return jsonify({"error": "document_id may contain only letters, numbers, dots, underscores, and hyphens."}), 400
        if document_id in documents:
            return jsonify({"error": f"Document {document_id!r} already exists."}), 409
        suffix = Path(filename).suffix.lower()
        stored_path = uploads_path / f"{document_id}{suffix}"
        stored_path.parent.mkdir(parents=True, exist_ok=True)
        stored_path.write_bytes(content)
        now = datetime.now(timezone.utc).isoformat()
        document = {
            "document_id": document_id,
            "filename": filename,
            "file_type": suffix[1:],
            "stored_filename": stored_path.name,
            "file_size": len(content),
            "text": text,
            "status": "stored",
            "entities": [],
            "candidate_relations": [],
            "created_at": now,
            "updated_at": now,
        }
        documents[document_id] = document
        persist_documents()
        return jsonify({"document": public_document(document)}), 201

    @app.get("/graph")
    def get_graph() -> Any:
        return jsonify(state_graph.export_json())

    @app.post("/graph/fact")
    def add_fact() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        required = ("subject", "predicate", "object")
        missing = [field for field in required if not isinstance(payload.get(field), str) or not payload[field].strip()]
        if missing:
            return jsonify({"error": f"Required non-empty fields: {', '.join(missing)}"}), 400
        try:
            relation = state_graph.add_fact(
                payload["subject"],
                payload["predicate"],
                payload["object"],
                source=payload.get("source", "api"),
                confidence=float(payload.get("confidence", 1.0)),
                relation_type=payload.get("relation_type", "observed"),
                provenance=payload.get("provenance", {}),
            )
        except (TypeError, ValueError) as error:
            return jsonify({"error": str(error)}), 400
        persist_graph()
        return jsonify({"status": "created", "relation": relation.to_dict()}), 201

    @app.post("/graph/review")
    def review_candidate() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        if payload.get("decision") != "approve":
            return jsonify({"status": "rejected", "inserted": False}), 200
        required = ("subject", "predicate", "object", "document_id", "evidence_snippet")
        missing = [field for field in required if not isinstance(payload.get(field), str) or not payload[field].strip()]
        if missing:
            return jsonify({"error": f"Required non-empty fields: {', '.join(missing)}"}), 400
        relation = state_graph.add_fact(
            payload["subject"],
            payload["predicate"],
            payload["object"],
            source=payload["document_id"],
            relation_type="human_reviewed",
            provenance={
                "document_id": payload["document_id"],
                "snippet": payload["evidence_snippet"],
                "reviewed_by": payload.get("reviewed_by", "api-user"),
            },
        )
        persist_graph()
        return jsonify({"status": "approved", "inserted": True, "relation": relation.to_dict()}), 201

    @app.post("/documents/<document_id>/review")
    def review_document_candidate(document_id: str) -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        document = documents.get(document_id)
        if document is None:
            return jsonify({"error": "Document not found."}), 404
        try:
            candidate_index = int(payload.get("candidate_index"))
            candidate = document["candidate_relations"][candidate_index]
        except (TypeError, ValueError, IndexError):
            return jsonify({"error": "candidate_index must identify an extracted candidate."}), 400
        decision = payload.get("decision")
        if decision not in {"approve", "reject"}:
            return jsonify({"error": "decision must be `approve` or `reject`."}), 400
        if candidate.get("status") != "extracted_candidate":
            return jsonify({"error": "Candidate has already been reviewed."}), 409
        if decision == "approve":
            relation = state_graph.add_fact(
                candidate["subject"],
                candidate["label"],
                candidate["object"],
                source=document_id,
                confidence=float(candidate.get("score", 1.0)),
                relation_type="human_reviewed",
                provenance={
                    "document_id": document_id,
                    "snippet": candidate["evidence_snippet"],
                    "source_offsets": candidate.get("source_offsets", {}),
                    "reviewed_by": payload.get("reviewed_by", "api-user"),
                },
            )
            persist_graph()
            candidate["reviewed_relation"] = relation.to_dict()
            candidate["status"] = "human_reviewed"
        else:
            candidate["status"] = "rejected"
        document["updated_at"] = datetime.now(timezone.utc).isoformat()
        document["status"] = "review_required" if any(item["status"] == "extracted_candidate" for item in document["candidate_relations"]) else "reviewed"
        persist_documents()
        return jsonify({"document": public_document(document), "inserted": decision == "approve"})

    @app.post("/nlp/dataset/validate")
    def validate_dataset() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"status": "invalid", "error": "Request body must be a JSON object."}), 400
        try:
            pipeline = NLPTrainingPipeline({**nlp_config, "dataset_path": payload.get("dataset_path") or nlp_config.get("dataset_path")})
            data = pipeline.load_dataset()
            task = payload.get("task", "relation")
            if task not in {"relation", "ner"}:
                raise ValueError("Task must be `relation` or `ner`.")
            return jsonify({
                "status": "valid",
                "task": task,
                "examples": len(data),
                "splits": pipeline.validate_dataset(data, task),
                "synthetic_data": pipeline.synthetic_data,
            })
        except (TypeError, ValueError, FileNotFoundError) as error:
            return jsonify({"status": "invalid", "error": str(error)}), 400

    @app.post("/gnn/dataset/validate")
    def validate_gnn_dataset() -> Any:
        try:
            trainer = GraphNeuralNetworkTrainer(state_graph, gnn_config)
            dataset = trainer.prepare_dataset()
            return jsonify({
                "status": "valid" if dataset["splits"]["train"] and dataset["splits"]["validation"] and dataset["splits"]["test"] else "insufficient_data",
                "entities": len(dataset["entity_to_id"]),
                "relations": len(dataset["relation_to_id"]),
                "split_sizes": {name: len(rows) for name, rows in dataset["splits"].items()},
                "message_passing_edges": len(dataset["message_passing_edges"]),
                "split_source": dataset["split_source"],
            })
        except (ValueError, RuntimeError) as error:
            return jsonify({"status": "invalid", "error": str(error)}), 400

    @app.post("/nlp/train")
    def train_nlp() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"status": "invalid", "error": "Request body must be a JSON object."}), 400
        task = payload.get("task", "relation")
        try:
            config = payload.get("config", {})
            if not isinstance(config, dict):
                raise ValueError("config must be a JSON object.")
            pipeline = NLPTrainingPipeline(
                {
                    **nlp_config,
                    **config,
                    "dataset_path": payload.get("dataset_path") or nlp_config.get("dataset_path"),
                }
            )
            dataset = pipeline.load_dataset()
            splits = pipeline.validate_dataset(dataset, task)
        except (TypeError, ValueError, FileNotFoundError) as error:
            return jsonify({"status": "invalid", "error": str(error)}), 400
        job_id = start_job(
            f"nlp-{task}",
            lambda progress: pipeline.train(dataset, task=task, progress_callback=progress),
        )
        return jsonify({"status": "queued", "job_id": job_id, "splits": splits, "synthetic_data": pipeline.synthetic_data}), 202

    @app.post("/nlp/demo-setup")
    def setup_demo_models() -> Any:
        dataset_path = Path(__file__).resolve().parent.parent / "data" / "sample_nlp_dataset.json"
        if not dataset_path.is_file():
            return jsonify({"status": "unavailable", "error": f"Bundled demo dataset is missing: {dataset_path}"}), 500
        try:
            demo_pipeline = NLPTrainingPipeline({
                **nlp_config,
                "dataset_path": str(dataset_path),
                "epochs": 20,
                "learning_rate": 1e-2,
                "ner_architecture": "local-context",
            })
            examples = demo_pipeline.load_dataset()
            demo_pipeline.validate_dataset(examples, "ner")
            demo_pipeline.validate_dataset(examples, "relation")
            relation_pipeline = NLPTrainingPipeline({
                **nlp_config,
                "dataset_path": str(dataset_path),
                "epochs": 20,
                "learning_rate": 1e-3,
                "synthetic_data": demo_pipeline.synthetic_data,
            })
        except (TypeError, ValueError, FileNotFoundError) as error:
            return jsonify({"status": "invalid", "error": str(error)}), 400

        def run(progress: Any) -> Dict[str, Any]:
            ner_result = demo_pipeline.train(examples, task="ner", progress_callback=lambda text: progress(f"NER {text}"))
            relation_result = relation_pipeline.train(
                examples,
                task="relation",
                progress_callback=lambda text: progress(f"Relation {text}"),
            )
            return {
                "status": "completed",
                "task": "demo-extraction-models",
                "checkpoints": {
                    "ner": ner_result["checkpoint"],
                    "relation": relation_result["checkpoint"],
                },
                "synthetic_data": True,
                "warning": "Both models were genuinely trained on the bundled synthetic smoke dataset. They are for workflow demonstration only, not reliable document extraction or benchmark performance.",
                "ner": {
                    "train_loss": ner_result["train_loss"],
                    "evaluation": ner_result["evaluation"],
                },
                "relation": {
                    "train_loss": relation_result["train_loss"],
                    "evaluation": relation_result["evaluation"],
                },
            }

        job_id = start_job("nlp-demo-setup", run)
        return jsonify({"status": "queued", "job_id": job_id, "dataset": str(dataset_path), "synthetic_data": True}), 202

    @app.post("/gnn/train")
    def train_gnn() -> Any:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"status": "invalid", "error": "Request body must be a JSON object."}), 400
        config = payload.get("config", {})
        if not isinstance(config, dict):
            return jsonify({"status": "invalid", "error": "config must be a JSON object."}), 400
        try:
            trainer = GraphNeuralNetworkTrainer(state_graph, {**gnn_config, **config})
        except (TypeError, ValueError, RuntimeError) as error:
            return jsonify({"status": "invalid", "error": str(error)}), 400
        job_id = start_job("gnn-rgcn", lambda progress: trainer.train(progress_callback=progress))
        return jsonify({"status": "queued", "job_id": job_id}), 202

    @app.post("/gnn/predict")
    def predict_gnn() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        required = ("checkpoint", "subject", "predicate")
        if any(not payload.get(field) for field in required):
            return jsonify({"error": "`checkpoint`, `subject`, and `predicate` are required."}), 400
        try:
            checkpoint = checkpoint_path(payload["checkpoint"], gnn_config["checkpoint_dir"])
            trainer = GraphNeuralNetworkTrainer.load_checkpoint(checkpoint, state_graph)
            candidates = trainer.predict(payload["subject"], payload["predicate"], int(payload.get("top_k", 5)))
            return jsonify({"candidates": candidates, "inserted_into_graph": False})
        except (OSError, ValueError, RuntimeError) as error:
            return jsonify({"error": str(error)}), 400

    @app.get("/jobs/<job_id>")
    def get_job(job_id: str) -> Any:
        with lock:
            job = jobs.get(job_id)
            if job is None:
                saved = tracker.root / f"{job_id}.json"
                if saved.exists():
                    return jsonify(json.loads(saved.read_text(encoding="utf-8")))
                return jsonify({"error": "Job not found."}), 404
            return jsonify({key: value for key, value in job.items() if key not in {"future", "retry_run"}})

    @app.get("/experiments")
    def list_experiments() -> Any:
        items = []
        for path in tracker.list():
            items.append(json.loads(path.read_text(encoding="utf-8")))
        items.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return jsonify({"experiments": items})

    @app.post("/jobs/<job_id>/cancel")
    def cancel_job(job_id: str) -> Any:
        with lock:
            job = jobs.get(job_id)
            if job is None:
                return jsonify({"error": "Job not found."}), 404
            future: Future = job["future"]
            if not future.cancel():
                return jsonify({"error": "Job is already running or finished; running model training cannot be safely interrupted."}), 409
            job["status"] = "cancelled"
            job["logs"].append("Queued training was cancelled.")
            persist(job_id)
            return jsonify({key: value for key, value in job.items() if key not in {"future", "retry_run"}})

    @app.post("/jobs/<job_id>/retry")
    def retry_job(job_id: str) -> Any:
        with lock:
            previous = jobs.get(job_id)
            if previous is None:
                return jsonify({"error": "Job is not available in this process; submit the same task and configuration again."}), 409
            if previous["status"] != "failed":
                return jsonify({"error": "Only failed jobs can be retried."}), 409
            run = previous.get("retry_run")
            task = previous["task"]
        if run is None:
            return jsonify({"error": "Retry configuration is unavailable; submit a new training request."}), 409
        retry_id = start_job(task, run, retry_of=job_id)
        return jsonify({"status": "queued", "job_id": retry_id, "retry_of": job_id}), 202

    @app.post("/nlp/infer")
    def infer_nlp() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        if not isinstance(payload.get("text"), str) or not payload["text"].strip() or not payload.get("checkpoint"):
            return jsonify({"error": "`text` and `checkpoint` are required."}), 400
        try:
            task = payload.get("task", "relation")
            checkpoint = checkpoint_path(payload["checkpoint"], nlp_pipeline.checkpoint_dir)
            if task == "ner":
                model = EntityRecognizer(checkpoint)
                return jsonify({"entities": model.infer(payload["text"]), "checkpoint": str(checkpoint)})
            if task == "relation":
                model = RelationExtractor(checkpoint)
                result = model.predict(payload["text"], payload.get("subject", {}), payload.get("object", {}))
                return jsonify({**result, "checkpoint": str(checkpoint)})
            return jsonify({"error": "Task must be `ner` or `relation`."}), 400
        except (OSError, ValueError, RuntimeError) as error:
            return jsonify({"error": str(error)}), 400

    @app.post("/documents/extract")
    def extract_document() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        document_id = payload.get("document_id")
        if not isinstance(document_id, str) or not document_id.strip():
            return jsonify({"error": "`document_id` is required."}), 400
        document = documents.get(document_id)
        if document is None:
            return jsonify({"error": "Upload or store the document before extraction."}), 404
        supplied_text = payload.get("text")
        if supplied_text is not None and supplied_text != document.get("text"):
            return jsonify({"error": "Extraction text must match the text stored for this document."}), 409
        text = document["text"]
        try:
            missing_checkpoints = [name for name in ("ner_checkpoint", "relation_checkpoint") if not payload.get(name)]
            if missing_checkpoints:
                available = nlp_checkpoint_catalog()
                available_tasks = sorted({item.get("task") for item in available if item.get("compatible")})
                missing_tasks = [task for task in ("ner", "relation") if task not in available_tasks]
                missing_labels = [("NER" if task == "ner" else "relation") for task in missing_tasks]
                raise ValueError(
                    f"Select a compatible NER and relation checkpoint. Missing compatible model(s): "
                    f"{', '.join(missing_labels) or ', '.join(missing_checkpoints)}. "
                    "Train them in Model Lab or use the clearly labeled demo setup."
                )
            ner_checkpoint = checkpoint_path(payload["ner_checkpoint"], nlp_pipeline.checkpoint_dir)
            relation_checkpoint = checkpoint_path(payload["relation_checkpoint"], nlp_pipeline.checkpoint_dir)
            ner_metadata = json.loads((ner_checkpoint / "metadata.json").read_text(encoding="utf-8"))
            relation_metadata = json.loads((relation_checkpoint / "metadata.json").read_text(encoding="utf-8"))
            if ner_metadata.get("task") != "ner" or relation_metadata.get("task") != "relation":
                raise ValueError("Selected checkpoints are not a compatible NER checkpoint and relation checkpoint.")
            recognizer = EntityRecognizer(ner_checkpoint)
            classifier = RelationExtractor(relation_checkpoint)
            chunks = _text_chunks(text)
            entities = []
            candidates = []
            pair_budget_per_chunk = int(settings.get("max_candidate_pairs_per_chunk", 256))
            if pair_budget_per_chunk < 1:
                raise ValueError("max_candidate_pairs_per_chunk must be positive.")
            pairs_scored = 0
            pairs_skipped_by_budget = 0
            for chunk_start, chunk in chunks:
                chunk_pairs_scored = 0
                chunk_entities = recognizer.infer(chunk)
                for entity in chunk_entities:
                    entities.append({
                        **entity,
                        "start": entity["start"] + chunk_start,
                        "end": entity["end"] + chunk_start,
                    })
                for subject in chunk_entities:
                    for obj in chunk_entities:
                        if subject is obj or (
                            subject["start"] < obj["end"] and obj["start"] < subject["end"]
                        ):
                            continue
                        if chunk_pairs_scored >= pair_budget_per_chunk:
                            pairs_skipped_by_budget += 1
                            continue
                        chunk_pairs_scored += 1
                        pairs_scored += 1
                        prediction = classifier.predict(
                            chunk,
                            {"start": subject["start"], "end": subject["end"]},
                            {"start": obj["start"], "end": obj["end"]},
                        )
                        if prediction["label"].lower() in {"no_relation", "not_a_relation", "none"}:
                            continue
                        candidates.append({
                            **prediction,
                            "subject_label": subject["label"],
                            "object_label": obj["label"],
                            "synthetic_model": bool(
                                ner_metadata.get("synthetic_data") or relation_metadata.get("synthetic_data")
                            ),
                            "status": "extracted_candidate",
                            "document_id": document_id,
                            "evidence_snippet": chunk.strip(),
                            "source_offsets": {
                                "start": chunk_start + min(subject["start"], obj["start"]),
                                "end": chunk_start + max(subject["end"], obj["end"]),
                            },
                            "subject_offsets": {"start": subject["start"] + chunk_start, "end": subject["end"] + chunk_start},
                            "object_offsets": {"start": obj["start"] + chunk_start, "end": obj["end"] + chunk_start},
                            "requires_human_review": True,
                        })
            now = datetime.now(timezone.utc).isoformat()
            document.update({
                "entities": entities,
                "candidate_relations": candidates,
                "status": "review_required" if candidates else "extracted",
                "extraction": {
                    "ner_checkpoint": str(ner_checkpoint),
                    "relation_checkpoint": str(relation_checkpoint),
                    "synthetic_models": bool(
                        ner_metadata.get("synthetic_data") or relation_metadata.get("synthetic_data")
                    ),
                    "chunks_processed": len(chunks),
                    "entity_count": len(entities),
                    "candidate_count": len(candidates),
                    "pairs_scored": pairs_scored,
                    "pairs_skipped_by_budget": pairs_skipped_by_budget,
                    "coverage_note": "Model output is a best-effort candidate extraction, not a guarantee of all facts in the document.",
                },
                "updated_at": now,
            })
            documents[document_id] = document
            persist_documents()
            return jsonify({"entities": entities, "candidate_relations": candidates, "inserted_into_graph": False})
        except (KeyError, OSError, ValueError, RuntimeError) as error:
            return jsonify({"error": str(error)}), 400

    @app.post("/verify")
    def verify_claim() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        try:
            return jsonify(verifier.verify(payload.get("claim", "")))
        except (OSError, RuntimeError, ValueError) as error:
            return jsonify({"error": str(error)}), 400

    @app.post("/verify/dataset/validate")
    def validate_verifier_dataset() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"status": "invalid", "error": "Request body must be a JSON object."}), 400
        try:
            config = {
                **nli_config,
                "dataset_path": payload.get("dataset_path") or nli_config.get("dataset_path"),
            }
            trainer = ClaimNLITrainer(config)
            rows = trainer.load_dataset()
            splits = trainer._split(rows)
            return jsonify({
                "status": "valid",
                "task": "claim_nli",
                "examples": len(rows),
                "splits": {name: len(items) for name, items in splits.items()},
                "labels": {label: sum(row["label"] == label for row in rows) for label in sorted({row["label"] for row in rows})},
                "synthetic_data": trainer.synthetic_data,
            })
        except (TypeError, ValueError, FileNotFoundError) as error:
            return jsonify({"status": "invalid", "error": str(error)}), 400

    @app.post("/verify/train")
    def train_verifier() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"status": "invalid", "error": "Request body must be a JSON object."}), 400
        try:
            config_values = payload.get("config", {})
            if not isinstance(config_values, dict):
                raise ValueError("config must be a JSON object.")
            config = {
                **nli_config,
                **config_values,
                "dataset_path": payload.get("dataset_path") or nli_config.get("dataset_path"),
            }
            trainer = ClaimNLITrainer(config)
            rows = trainer.load_dataset()
            splits = trainer._split(rows)
        except (TypeError, ValueError, FileNotFoundError) as error:
            return jsonify({"status": "invalid", "error": str(error)}), 400

        def run(progress: Any) -> Dict[str, Any]:
            result = trainer.train(rows, progress_callback=progress)
            verifier.nli_classifier = TransformerNLIClassifier(result["checkpoint"])
            return result

        job_id = start_job("claim-nli", run)
        return jsonify({"status": "queued", "job_id": job_id, "split_sizes": {key: len(rows) for key, rows in splits.items()}}), 202

    @app.post("/rag")
    def retrieve_rag() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        query = payload.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return jsonify({"error": "A non-empty query is required."}), 400
        return jsonify(rag.answer(query))

    return app
