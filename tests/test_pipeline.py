import json
import time
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile
from pathlib import Path

import pytest
import torch

from knowledge_graph_builder import (
    ClaimVerifier,
    GraphNeuralNetworkTrainer,
    GraphRAGEngine,
    KnowledgeGraph,
    NLPTrainingPipeline,
)
from knowledge_graph_builder.app import create_app
from knowledge_graph_builder.dataset_prep import (
    combine_datasets,
    load_kg_benchmark,
    prepare_conll2003,
    prepare_docred,
    prepare_fever,
)
from knowledge_graph_builder.document_ingest import extract_document_text
from knowledge_graph_builder.gnn import _make_model, negative_sample, ranking_metrics
from knowledge_graph_builder.nlp import DatasetExample, RelationExtractor, _RelationClassifier, classification_metrics, split_dataset
from knowledge_graph_builder.verifier import ClaimNLITrainer, TransformerNLIClassifier


def relation_examples():
    examples = []
    pairs = [
        ("Alice works at Acme.", (0, 5), (15, 19), "works_at"),
        ("Bob lives in Paris.", (0, 3), (13, 18), "located_in"),
    ]
    for index in range(16):
        text, subject, obj, label = pairs[index % 2]
        prefix = f"Person{index} "
        suffix = f" Company{index}." if label == "works_at" else f" City{index}."
        phrase = "works at" if label == "works_at" else "lives in"
        text = f"Person{index} {phrase}{suffix}"
        entity = f"Company{index}" if label == "works_at" else f"City{index}"
        entity_start = text.index(entity)
        examples.append(
            DatasetExample(
                text=text,
                relations=[
                    {
                        "subject": {"start": 0, "end": len(f"Person{index}")},
                        "object": {"start": entity_start, "end": entity_start + len(entity)},
                        "label": label,
                    }
                ],
            )
        )
    return examples


def make_graph(edge_count=24):
    graph = KnowledgeGraph("gnn-test")
    for index in range(12):
        graph.add_entity(f"entity-{index}", "entity")
    edges = []
    for index in range(edge_count // 2):
        edges.append((index % 12, (index + 1) % 12, "connected_to"))
        edges.append((index % 12, (index + 4) % 12, "associated_with"))
    for source, target, relation in edges:
        graph.add_fact(f"entity-{source}", relation, f"entity-{target}")
    return graph


def test_graph_round_trip_preserves_provenance(tmp_path):
    graph = KnowledgeGraph("demo")
    graph.add_entity("alice", "person", {"department": "research"})
    graph.add_fact(
        "alice",
        "works_at",
        "OpenAI",
        source="doc-17",
        relation_type="verified",
        provenance={"document_id": "doc-17", "snippet": "Alice works at OpenAI."},
    )
    target = tmp_path / "graph.json"
    graph.save_json(target)
    loaded = KnowledgeGraph.load_json(target)
    relation = loaded.query(subject="alice")[0]
    assert relation.source == "doc-17"
    assert relation.provenance == {"document_id": "doc-17", "snippet": "Alice works at OpenAI."}
    assert loaded.find_entity("alice").attributes == {"department": "research"}


def test_conll_preparation_preserves_official_splits_and_word_labels(tmp_path):
    source_files = {}
    for split in ("train", "validation", "test"):
        source = tmp_path / f"{split}.txt"
        source.write_text(
            "Ada NNP B-NP B-PER\nworks VBZ B-VP O\n\nAcme NNP B-NP B-ORG\n\n",
            encoding="utf-8",
        )
        output = tmp_path / f"{split}.json"
        prepare_conll2003(source, output, split)
        source_files[split] = output
    combined = tmp_path / "conll.json"
    combine_datasets(source_files, combined, "CoNLL-2003")
    pipeline = NLPTrainingPipeline({"dataset_path": str(combined), "random_seed": 12})
    examples = pipeline.load_dataset()
    assert pipeline.validate_dataset(examples, "ner") == {"train": 2, "validation": 2, "test": 2}
    assert examples[0].ner_tokens == ["Ada", "works"]
    assert examples[0].ner_tags == ["B-PER", "O"]
    assert all(row.metadata["split"] == split for split, rows in split_dataset(examples).items() for row in rows)


def test_docred_preparation_maps_relation_labels_and_evidence(tmp_path):
    source = tmp_path / "docred.json"
    source.write_text(json.dumps([{
        "title": "doc-1",
        "sents": [["Ada", "works", "at", "Acme", "."]],
        "vertexSet": [
            [{"name": "Ada", "sent_id": 0, "pos": [0, 1]}],
            [{"name": "Acme", "sent_id": 0, "pos": [3, 4]}],
        ],
        "labels": [{"h": 0, "t": 1, "r": "P1", "evidence": [0]}],
    }]), encoding="utf-8")
    relation_map = tmp_path / "rel_info.json"
    relation_map.write_text(json.dumps({"P1": {"title": "works_at"}}), encoding="utf-8")
    output = prepare_docred(source, tmp_path / "prepared.json", relation_map, "train")
    row = json.loads(output.read_text(encoding="utf-8"))["examples"][0]
    annotation = row["relations"][0]
    assert row["text"] == "Ada works at Acme ."
    assert row["metadata"]["split"] == "train"
    assert annotation["label"] == "works_at"
    assert row["text"][annotation["subject"]["start"]:annotation["subject"]["end"]] == "Ada"
    assert row["text"][annotation["object"]["start"]:annotation["object"]["end"]] == "Acme"
    assert annotation["evidence"] == ["Ada works at Acme ."]


def test_document_upload_text_and_docx_extraction():
    assert extract_document_text("notes.txt", b"\xef\xbb\xbfAda works at Acme.") == "Ada works at Acme."
    docx = BytesIO()
    with ZipFile(docx, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body><w:p><w:r><w:t>Ada works at Acme.</w:t></w:r></w:p>'
            '<w:p><w:r><w:t>She leads research.</w:t></w:r></w:p></w:body></w:document>',
        )
    assert extract_document_text("source.docx", docx.getvalue()) == "Ada works at Acme.\nShe leads research."
    with pytest.raises(ValueError, match="Unsupported document type"):
        extract_document_text("source.exe", b"no")
    with pytest.raises(ValueError, match="UTF-8"):
        extract_document_text("source.txt", b"\xff")
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    pdf_content = BytesIO()
    writer.write(pdf_content)
    assert extract_document_text("scan.pdf", pdf_content.getvalue()) == ""


def test_uploaded_document_is_chunk_extracted_reviewed_and_saved_to_graph(tmp_path, monkeypatch):
    import knowledge_graph_builder.app as app_module

    calls = []

    class FakeRecognizer:
        def __init__(self, _checkpoint):
            pass

        def infer(self, chunk):
            calls.append(chunk)
            results = []
            for name in ("Ada", "Acme"):
                start = chunk.find(name)
                if start >= 0:
                    results.append({"text": name, "label": "ENTITY", "start": start, "end": start + len(name)})
            return results

    class FakeRelationExtractor:
        def __init__(self, _checkpoint):
            pass

        def predict(self, chunk, subject, obj):
            subject_text = chunk[subject["start"]:subject["end"]]
            object_text = chunk[obj["start"]:obj["end"]]
            return {
                "subject": subject_text,
                "object": object_text,
                "label": "works_at" if subject_text == "Ada" and object_text == "Acme" else "NO_RELATION",
                "score": 0.9,
            }

    monkeypatch.setattr(app_module, "EntityRecognizer", FakeRecognizer)
    monkeypatch.setattr(app_module, "RelationExtractor", FakeRelationExtractor)
    checkpoint_dir = tmp_path / "nlp"
    checkpoint_dir.mkdir()
    ner_checkpoint = checkpoint_dir / "ner-test"
    relation_checkpoint = checkpoint_dir / "relation-test"
    ner_checkpoint.mkdir()
    relation_checkpoint.mkdir()
    (ner_checkpoint / "metadata.json").write_text(json.dumps({"task": "ner"}), encoding="utf-8")
    (relation_checkpoint / "metadata.json").write_text(json.dumps({"task": "relation"}), encoding="utf-8")
    text = ("Ada works at Acme. " + "A" * 1300 + " ") + "Ada works at Acme."
    graph_path = tmp_path / "graph.json"
    app = create_app(KnowledgeGraph("upload-test"), {
        "graph_path": str(graph_path),
        "documents_path": str(tmp_path / "documents.json"),
        "uploads_path": str(tmp_path / "uploads"),
        "experiment_dir": str(tmp_path / "experiments"),
        "nlp_checkpoint_dir": str(checkpoint_dir),
    })
    client = app.test_client()
    uploaded = client.post(
        "/documents/upload",
        data={"document_id": "doc-upload", "file": (BytesIO(text.encode("utf-8")), "../../source.txt")},
        content_type="multipart/form-data",
    )
    assert uploaded.status_code == 201
    document = uploaded.json["document"]
    assert document["filename"] == "source.txt"
    assert document["text_length"] == len(text)
    assert document["preview"] == text[:240]
    assert (tmp_path / "uploads" / "doc-upload.txt").exists()
    extracted = client.post("/documents/extract", json={
        "document_id": "doc-upload",
        "ner_checkpoint": str(ner_checkpoint),
        "relation_checkpoint": str(relation_checkpoint),
    })
    assert extracted.status_code == 200
    assert extracted.json["inserted_into_graph"] is False
    assert extracted.json["candidate_relations"][0]["label"] == "works_at"
    assert extracted.json["candidate_relations"][0]["source_offsets"] == {"start": 0, "end": 17}
    assert len(calls) >= 2
    assert extracted.json["candidate_relations"][-1]["source_offsets"]["start"] > 1200
    reviewed = client.post("/documents/doc-upload/review", json={"candidate_index": 0, "decision": "approve"})
    assert reviewed.status_code == 200
    assert reviewed.json["inserted"] is True
    relation = client.get("/graph").json["relations"][0]
    assert relation["relation_type"] == "human_reviewed"
    assert relation["provenance"]["document_id"] == "doc-upload"


def test_document_upload_rejects_unsupported_invalid_and_oversized_files(tmp_path):
    app = create_app(KnowledgeGraph("upload-validation"), {
        "documents_path": str(tmp_path / "documents.json"),
        "uploads_path": str(tmp_path / "uploads"),
        "experiment_dir": str(tmp_path / "experiments"),
        "max_upload_bytes": 1024,
    })
    client = app.test_client()
    unsupported = client.post("/documents/upload", data={
        "file": (BytesIO(b"content"), "payload.exe"),
    }, content_type="multipart/form-data")
    assert unsupported.status_code == 400
    invalid_id = client.post("/documents/upload", data={
        "document_id": "../outside",
        "file": (BytesIO(b"text"), "payload.txt"),
    }, content_type="multipart/form-data")
    assert invalid_id.status_code == 400
    oversized = client.post("/documents/upload", data={
        "file": (BytesIO(b"x" * 1100), "large.txt"),
    }, content_type="multipart/form-data")
    assert oversized.status_code == 413
    assert not (tmp_path / "outside.txt").exists()


def test_fever_preparation_and_separate_evidence_retrieval_evaluation(tmp_path):
    source = tmp_path / "fever.jsonl"
    source.write_text(
        "\n".join([
            json.dumps({
                "id": 1,
                "claim": "Ada works at Acme.",
                "label": "SUPPORTS",
                "evidence": [[[None, None, "Ada_Page", 0]]],
            }),
            json.dumps({"id": 2, "claim": "A distant unrelated assertion.", "label": "NOT ENOUGH INFO", "evidence": []}),
        ]),
        encoding="utf-8",
    )
    prepared = prepare_fever(source, tmp_path / "fever-prepared.json")
    examples = json.loads(prepared.read_text(encoding="utf-8"))["examples"]
    assert examples[0]["label"] == "SUPPORTED"
    assert examples[0]["evidence_groups"][0][0]["page"] == "Ada_Page"
    graph = KnowledgeGraph()
    graph.add_fact(
        "Ada", "works_at", "Acme", source="Ada_Page",
        provenance={"document_id": "Ada_Page", "snippet": "Ada works at Acme."},
    )
    report = ClaimVerifier(graph).evaluate_fever(examples)
    assert report["classification"]["accuracy"] == 1.0
    assert report["classification"]["macro_f1"] == 1.0
    assert report["evidence_retrieval"]["precision"] == 1.0
    assert report["evidence_retrieval"]["recall"] == 1.0
    assert report["nei_retrieval_rate"] == 0.0


def test_standard_kg_split_training_checkpoint_and_filtered_head_tail_ranking(tmp_path):
    data = tmp_path / "WN18RR"
    data.mkdir()
    (data / "train.txt").write_text(
        "a\tr\tb\nb\tr\tc\nc\tr\td\nd\tr\te\ne\tr\tf\nf\tr\ta\na\tr\tc\nb\tr\td\nc\tr\te\n",
        encoding="utf-8",
    )
    (data / "valid.txt").write_text("f\tr\tg\n", encoding="utf-8")
    (data / "test.txt").write_text("g\tr\ta\n", encoding="utf-8")
    graph, benchmark_splits = load_kg_benchmark(data)
    trainer = GraphNeuralNetworkTrainer(graph, {
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "epochs": 1,
        "hidden_dim": 8,
        "device": "cpu",
        "benchmark_splits": benchmark_splits,
        "dataset_name": "WN18RR-test-fixture",
    })
    result = trainer.train()
    assert result["status"] == "completed"
    assert result["split_sizes"] == {"train": 9, "validation": 1, "test": 1}
    assert result["query_count"] == 2
    assert result["evaluation_protocol"].startswith("filtered tail and head")
    assert set(trainer.split["train"]).isdisjoint(trainer.split["test"])
    restored = GraphNeuralNetworkTrainer.load_checkpoint(result["checkpoint"], graph)
    assert restored.evaluate_checkpoint()["query_count"] == 2
    assert restored.evaluate_checkpoint()["metrics"] == result["metrics"]


def test_frontend_and_document_api_persist_local_state(tmp_path):
    settings = {
        "documents_path": str(tmp_path / "documents.json"),
        "graph_path": str(tmp_path / "graph.json"),
        "experiment_dir": str(tmp_path / "experiments"),
        "dataset_dir": str(tmp_path / "datasets"),
        "nlp_checkpoint_dir": str(tmp_path / "nlp"),
        "gnn_checkpoint_dir": str(tmp_path / "gnn"),
        "nli_checkpoint_dir": str(tmp_path / "nli"),
    }
    app = create_app(
        KnowledgeGraph("workspace"),
        settings,
    )
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert b"KnowGraph AI" in client.get("/").data
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    created = client.post("/documents", json={"document_id": "doc-local", "text": "Ada works at Acme."})
    assert created.status_code == 201
    assert created.json["document"]["status"] == "stored"
    assert client.get("/overview").json["documents"] == 1
    client.post("/graph/fact", json={
        "subject": "Ada", "predicate": "works_at", "object": "Acme",
        "source": "doc-local", "provenance": {"document_id": "doc-local", "snippet": "Ada works at Acme."},
    })
    loaded_app = create_app(None, settings)
    loaded_client = loaded_app.test_client()
    assert loaded_client.get("/documents").json["documents"][0]["document_id"] == "doc-local"
    assert loaded_client.get("/graph").json["relations"][0]["provenance"]["document_id"] == "doc-local"


def test_model_lab_dataset_validation_routes_report_errors_and_splits(tmp_path):
    sample_dataset = Path(__file__).parent.parent / "data" / "sample_nlp_dataset.json"
    graph = KnowledgeGraph("tiny")
    graph.add_fact("a", "r", "b")
    graph.add_fact("b", "r", "c")
    app = create_app(graph, {
        "dataset_path": str(sample_dataset),
        "experiment_dir": str(tmp_path / "experiments"),
        "nlp_checkpoint_dir": str(tmp_path / "nlp"),
        "gnn_checkpoint_dir": str(tmp_path / "gnn"),
        "nli_checkpoint_dir": str(tmp_path / "nli"),
        "dataset_dir": str(tmp_path / "datasets"),
    })
    client = app.test_client()
    nlp_validation = client.post("/nlp/dataset/validate", json={"task": "relation"})
    assert nlp_validation.status_code == 200
    assert nlp_validation.json["splits"]["train"] > 0
    assert nlp_validation.json["synthetic_data"] is True
    invalid_nli = client.post("/verify/dataset/validate", json={"dataset_path": str(tmp_path / "missing.json")})
    assert invalid_nli.status_code == 400
    assert invalid_nli.json["status"] == "invalid"
    gnn_validation = client.post("/gnn/dataset/validate", json={})
    assert gnn_validation.status_code == 200
    assert gnn_validation.json["status"] == "insufficient_data"


def test_dataset_split_is_reproducible_disjoint_and_complete():
    examples = [DatasetExample(text=f"example {index}") for index in range(20)]
    first = split_dataset(examples, seed=19)
    second = split_dataset(examples, seed=19)
    assert first == second
    flattened = first["train"] + first["validation"] + first["test"]
    assert len(flattened) == len(examples)
    assert len({example.text for example in flattened}) == len(examples)
    assert all(first[key] for key in ("train", "validation", "test"))
    with pytest.raises(ValueError, match="At least 3"):
        split_dataset(examples[:2])


def test_relation_model_updates_weights_and_checkpoint_inference(tmp_path):
    examples = relation_examples()
    torch.manual_seed(42)
    before_model = _RelationClassifier(256, 2, embedding_dim=16)
    before = before_model.embedding.weight.detach().clone()
    pipeline = NLPTrainingPipeline(
        {
            "checkpoint_dir": str(tmp_path / "nlp"),
            "random_seed": 42,
            "epochs": 4,
            "batch_size": 4,
            "embedding_dim": 16,
            "synthetic_data": True,
        }
    )
    result = pipeline.train(examples, task="relation")
    assert result["status"] == "completed"
    assert len(result["train_loss"]) == 4
    assert len(result["validation_loss"]) == 4
    checkpoint = result["checkpoint"]
    metadata = json.loads((tmp_path / "nlp" / checkpoint.split("/")[-1] / "metadata.json").read_text())
    trained = torch.load(f"{checkpoint}/weights.pt", map_location="cpu", weights_only=True)
    assert not torch.equal(before[: len(trained["embedding.weight"])], trained["embedding.weight"])
    assert metadata["synthetic_data"] is True
    assert "not meaningful" in result["interpretation"]

    loaded = RelationExtractor(checkpoint)
    example = examples[0]
    prediction = loaded.predict(example.text, example.relations[0]["subject"], example.relations[0]["object"])
    assert prediction["label"] in {"works_at", "located_in"}
    assert 0.0 <= prediction["score"] <= 1.0
    assert prediction["subject"] == "Person0"
    assert pipeline.evaluate_relation_checkpoint(checkpoint, examples) == result["evaluation"]


def test_nlp_dataset_validation_requires_explicit_format(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"examples": [{"text": "No annotation."}]}), encoding="utf-8")
    pipeline = NLPTrainingPipeline({"dataset_path": str(invalid)})
    data = pipeline.load_dataset()
    with pytest.raises(ValueError, match="Relation training requires"):
        pipeline.validate_dataset(data, task="relation")
    with pytest.raises(FileNotFoundError, match="Provide `dataset_path`"):
        NLPTrainingPipeline().load_dataset()


def test_single_class_training_is_not_reported_as_success(tmp_path):
    examples = relation_examples()
    for example in examples:
        example.relations[0]["label"] = "works_at"
    pipeline = NLPTrainingPipeline({"checkpoint_dir": str(tmp_path), "epochs": 1})
    with pytest.raises(ValueError, match="at least two distinct"):
        pipeline.train(examples, task="relation")


def test_local_transformer_ner_training_evaluation_and_reload(tmp_path):
    from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

    model_dir = tmp_path / "tiny-transformer"
    model_dir.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "alice", "works", "at", "acme", "."]
    (model_dir / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(model_dir / "vocab.txt"), do_lower_case=True)
    tokenizer.save_pretrained(model_dir)
    id_to_label = {0: "B-ORG", 1: "B-PERSON", 2: "O"}
    model = BertForTokenClassification(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            num_labels=3,
            id2label=id_to_label,
            label2id={label: index for index, label in id_to_label.items()},
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
    )
    initial_classifier = model.classifier.weight.detach().clone()
    model.save_pretrained(model_dir)
    examples = [
        DatasetExample(
            text="alice works at acme.",
            ner_tokens=["alice", "works", "at", "acme."],
            ner_tags=["B-PERSON", "O", "O", "B-ORG"],
        )
        for _ in range(12)
    ]
    pipeline = NLPTrainingPipeline(
        {
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "pretrained_model": str(model_dir),
            "epochs": 1,
            "random_seed": 3,
        }
    )
    result = pipeline.train_ner(examples)
    assert result["status"] == "completed"
    assert result["train_loss"][0] > 0
    assert result["evaluation"]["support"] > 0
    saved = BertForTokenClassification.from_pretrained(result["checkpoint"], local_files_only=True)
    assert not torch.equal(initial_classifier, saved.classifier.weight.detach())
    from knowledge_graph_builder.nlp import EntityRecognizer

    inferred = EntityRecognizer(result["checkpoint"]).infer("alice works at acme.")
    assert inferred
    assert all(item["label"] in {"PERSON", "ORG"} for item in inferred)


def test_local_pytorch_ner_updates_weights_and_reloads_for_inference(tmp_path):
    from knowledge_graph_builder.nlp import EntityRecognizer, _LocalNERClassifier

    sample_dataset = Path(__file__).parent.parent / "data" / "sample_nlp_dataset.json"
    pipeline = NLPTrainingPipeline({
        "dataset_path": str(sample_dataset),
        "checkpoint_dir": str(tmp_path / "local-ner"),
        "random_seed": 42,
        "epochs": 3,
        "learning_rate": 0.01,
        "ner_architecture": "local-context",
    })
    examples = pipeline.load_dataset()
    result = pipeline.train(examples, task="ner")
    assert result["status"] == "completed"
    assert len(result["train_loss"]) == 3
    metadata = json.loads((Path(result["checkpoint"]) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["architecture"] == "local-context-feature-classifier"
    assert metadata["synthetic_data"] is True
    assert metadata["unknown_token_policy"] == "outside"

    torch.manual_seed(42)
    initial = _LocalNERClassifier(
        len(metadata["feature_to_id"]),
        len(metadata["id_to_label"]),
        metadata["embedding_dim"],
    ).state_dict()
    trained = torch.load(Path(result["checkpoint"]) / "weights.pt", map_location="cpu", weights_only=True)
    assert any(not torch.equal(initial[name], trained[name]) for name in initial)

    loaded = EntityRecognizer(result["checkpoint"])
    tokens = ["Alice", "works", "at", "Acme."]
    predictions = loaded.predict_tags(tokens)
    assert len(predictions) == len(tokens)
    assert EntityRecognizer(result["checkpoint"]).predict_tags(tokens) == predictions
    assert loaded.predict_tags(["Unseen"]) == ["O"]
    assert result["evaluation"] == pipeline.evaluate_ner(
        split_dataset(examples, 42)["test"],
        result["checkpoint"],
    )


def test_classification_metrics_known_predictions():
    result = classification_metrics(["supported", "supported", "refuted"], ["supported", "refuted", "refuted"])
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["per_class"]["supported"]["precision"] == pytest.approx(1.0)
    assert result["per_class"]["supported"]["recall"] == pytest.approx(0.5)
    assert result["macro_f1"] == pytest.approx(2 / 3)


def test_negative_sampling_excludes_known_positive_triples():
    positives = [(0, 0, 1), (1, 0, 2)]
    known = set(positives) | {(1, 0, 0)}
    negatives = negative_sample(positives, known, num_entities=5, count=4, seed=4)
    assert len(negatives) == 4
    assert len(set(negatives)) == 4
    assert not (set(negatives) & known)
    assert all(source != target for source, _, target in negatives)


def test_ranking_metrics_match_known_ranks():
    result = ranking_metrics([1, 2, 4])
    assert result["mrr"] == pytest.approx(7 / 12)
    assert result["hits@1"] == pytest.approx(1 / 3)
    assert result["hits@3"] == pytest.approx(2 / 3)
    assert result["hits@10"] == pytest.approx(1.0)


def test_filtered_link_ranking_excludes_other_known_true_heads_and_tails(tmp_path):
    graph = KnowledgeGraph()
    for entity in ("a", "b", "c", "d"):
        graph.add_entity(entity, entity)
    graph.add_fact("a", "r", "c")
    graph.add_fact("a", "r", "d")
    graph.add_fact("d", "r", "c")
    trainer = GraphNeuralNetworkTrainer(graph, {"checkpoint_dir": str(tmp_path)})
    trainer.known_triples = {(0, 0, 2), (0, 0, 3), (3, 0, 2)}

    class FixedScorer:
        relation_embedding = SimpleNamespace(weight=torch.tensor([[1.0]]))

    ranks = trainer._ranks(FixedScorer(), torch.tensor([[1.0], [2.0], [3.0], [4.0]]), [(0, 0, 2)])
    assert ranks == [1, 3]


def test_gnn_real_training_split_checkpoint_and_inference(tmp_path):
    graph = make_graph()
    trainer = GraphNeuralNetworkTrainer(
        graph,
        {"checkpoint_dir": str(tmp_path / "gnn"), "epochs": 3, "hidden_dim": 16, "random_seed": 12},
    )
    splits = trainer.prepare_dataset()["splits"]
    train_pairs = {tuple(sorted((source, target))) for source, _, target in splits["train"]}
    heldout_pairs = {
        tuple(sorted((source, target)))
        for key in ("validation", "test")
        for source, _, target in splits[key]
    }
    assert train_pairs.isdisjoint(heldout_pairs)

    torch.manual_seed(12)
    initial = _make_model(len(trainer.entity_to_id), len(trainer.relation_to_id), 16)
    initial_state = {key: value.detach().clone() for key, value in initial.state_dict().items()}
    result = trainer.train()
    assert result["status"] == "completed"
    assert len(result["train_loss"]) == 3
    assert len(result["validation_loss"]) == 3
    saved = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
    assert any(not torch.equal(initial_state[key], saved["state_dict"][key]) for key in initial_state)
    assert result["metrics"] == saved["metrics"]
    assert set(result["metrics"]) == {"mrr", "hits@1", "hits@3", "hits@10"}
    assert result["baseline"] == saved["baseline"]
    assert saved["config"]["inverse_relation_to_id"]

    loaded = GraphNeuralNetworkTrainer.load_checkpoint(result["checkpoint"], graph)
    assert loaded.evaluate_checkpoint()["metrics"] == result["metrics"]
    before_count = len(graph.relations)
    candidates = loaded.predict("entity-0", "connected_to", top_k=3)
    assert all(candidate["label"] == "predicted_candidate" for candidate in candidates)
    assert len(graph.relations) == before_count


def test_small_graph_reports_insufficient_data_without_metrics(tmp_path):
    graph = KnowledgeGraph("tiny")
    graph.add_fact("a", "r", "b")
    graph.add_fact("b", "r", "c")
    result = GraphNeuralNetworkTrainer(graph, {"checkpoint_dir": str(tmp_path)}).train()
    assert result["status"] == "insufficient_data"
    assert result["metrics"] is None
    assert result["checkpoint"] is None


def test_claim_verifier_needs_evidence_and_preserves_provenance():
    graph = KnowledgeGraph()
    graph.add_fact("Alice", "works_at", "Acme", source="report-4", confidence=0.75, provenance={"document_id": "report-4"})
    verifier = ClaimVerifier(graph)
    supported = verifier.verify("Alice works at Acme")
    refuted = verifier.verify("Alice does not work at Acme")
    unknown = verifier.verify("Alice works at OtherCorp")
    assert supported["label"] == "SUPPORTED"
    assert supported["evidence"][0]["provenance"]["document_id"] == "report-4"
    assert supported["evidence_strength"] == pytest.approx(0.75)
    assert supported["classifier_confidence"] is None
    assert refuted["label"] == "REFUTED"
    assert unknown["label"] == "NOT ENOUGH INFORMATION"


def test_local_transformer_nli_fine_tuning_checkpoint_and_claim_evidence(tmp_path):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    model_dir = tmp_path / "tiny-nli"
    model_dir.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "alice", "works", "at", "acme", "does", "not", "other", "."]
    (model_dir / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(model_dir / "vocab.txt"), do_lower_case=True)
    tokenizer.save_pretrained(model_dir)
    id_to_label = {0: "SUPPORTED", 1: "REFUTED", 2: "NOT ENOUGH INFORMATION"}
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            num_labels=3,
            id2label=id_to_label,
            label2id={label: index for index, label in id_to_label.items()},
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
    )
    initial_classifier = model.classifier.weight.detach().clone()
    model.save_pretrained(model_dir)
    labels = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION"] * 10
    examples = [
        {
            "premise": "Alice works at Acme.",
            "hypothesis": f"Alice statement {label.lower()}.",
            "label": label,
        }
        for label in labels
    ]
    pipeline = ClaimNLITrainer(
        {
            "pretrained_model": str(model_dir),
            "checkpoint_dir": str(tmp_path / "nli-checkpoints"),
            "epochs": 1,
            "batch_size": 4,
            "random_seed": 12,
            "synthetic_data": True,
        }
    )
    result = pipeline.train(examples)
    assert result["status"] == "completed"
    assert result["synthetic_data"] is True
    assert "not meaningful" in result["interpretation"]
    assert result["train_loss"][0] > 0
    assert result["evaluation"]["count"] == result["split_sizes"]["test"]
    saved = BertForSequenceClassification.from_pretrained(result["checkpoint"], local_files_only=True)
    assert not torch.equal(initial_classifier, saved.classifier.weight.detach())
    classifier = TransformerNLIClassifier(result["checkpoint"])
    assert classifier.predict("Alice works at Acme.", "Alice works at Acme.")["label"] in {
        "SUPPORTED",
        "REFUTED",
        "NOT ENOUGH INFORMATION",
    }

    graph = KnowledgeGraph()
    graph.add_fact(
        "Alice", "works_at", "Acme", source="doc-1", confidence=0.8,
        provenance={"snippet": "Alice works at Acme.", "document_id": "doc-1"},
    )
    verified = ClaimVerifier(graph, {"nli_checkpoint": result["checkpoint"]}).verify("Alice works at Acme.")
    assert verified["classifier_confidence"] is not None
    assert verified["evidence_strength"] == pytest.approx(0.8)
    assert verified["evidence"][0]["provenance"]["document_id"] == "doc-1"


def test_rag_separates_predicted_edges_from_source_evidence():
    graph = KnowledgeGraph()
    graph.add_fact("Alice", "works_at", "Acme", source="doc-1", provenance={"document_id": "doc-1"})
    graph.add_fact("Alice", "owns", "OtherCo", source="model", relation_type="predicted")
    answer = GraphRAGEngine(graph).answer("Alice works at Acme")
    assert len(answer["evidence"]) == 1
    assert answer["evidence"][0]["source"] == "doc-1"
    assert answer["predicted_candidates"] == []
    assert answer["evidence"][0]["provenance"]["document_id"] == "doc-1"


def test_api_reports_missing_dataset_and_validates_fact_payload(tmp_path):
    from knowledge_graph_builder.app import create_app

    app = create_app(KnowledgeGraph("api"), {"experiment_dir": str(tmp_path / "experiments")})
    client = app.test_client()
    response = client.post("/nlp/train", json={"task": "relation"})
    assert response.status_code == 400
    assert response.json["status"] == "invalid"
    assert client.post("/nlp/train", json={"task": "relation", "config": []}).status_code == 400
    assert client.post("/gnn/train", json={"config": []}).status_code == 400
    nli_response = client.post("/verify/train", json={})
    assert nli_response.status_code == 400
    invalid_fact = client.post("/graph/fact", json={"subject": "Alice"})
    assert invalid_fact.status_code == 400
    approved = client.post(
        "/graph/review",
        json={
            "decision": "approve",
            "subject": "Alice",
            "predicate": "works_at",
            "object": "Acme",
            "document_id": "doc-5",
            "evidence_snippet": "Alice works at Acme.",
        },
    )
    assert approved.status_code == 201
    assert approved.json["relation"]["relation_type"] == "human_reviewed"
    assert approved.json["relation"]["provenance"]["document_id"] == "doc-5"
    assert client.get("/experiments").json["experiments"] == []
    assert client.get("/jobs/missing").status_code == 404


def test_document_extraction_reports_missing_task_checkpoints(tmp_path):
    app = create_app(KnowledgeGraph("missing-models"), {
        "nlp_checkpoint_dir": str(tmp_path / "nlp"),
        "documents_path": str(tmp_path / "documents.json"),
        "uploads_path": str(tmp_path / "uploads"),
        "experiment_dir": str(tmp_path / "experiments"),
    })
    client = app.test_client()
    stored = client.post("/documents", json={"document_id": "doc-needs-models", "text": "Source text."})
    assert stored.status_code == 201
    response = client.post("/documents/extract", json={"document_id": "doc-needs-models"})
    assert response.status_code == 400
    assert "NER, relation" in response.json["error"]
    assert "Model Lab" in response.json["error"]


def test_demo_model_job_extracts_reviews_and_persists_provenance(tmp_path):
    app = create_app(KnowledgeGraph("demo-workflow"), {
        "nlp_checkpoint_dir": str(tmp_path / "nlp"),
        "graph_path": str(tmp_path / "graph.json"),
        "documents_path": str(tmp_path / "documents.json"),
        "uploads_path": str(tmp_path / "uploads"),
        "experiment_dir": str(tmp_path / "experiments"),
        "training_workers": 1,
    })
    client = app.test_client()
    response = client.post("/nlp/demo-setup", json={})
    assert response.status_code == 202
    job_id = response.json["job_id"]
    deadline = time.monotonic() + 15
    job = client.get(f"/jobs/{job_id}").json
    while job["status"] in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(0.02)
        job = client.get(f"/jobs/{job_id}").json
    assert job["status"] == "completed", job.get("error")
    assert job["result"]["synthetic_data"] is True
    assert "not reliable" in job["result"]["warning"]

    models = client.get("/models").json
    assert len(models["nlp_by_task"]["ner"]) == 1
    assert len(models["nlp_by_task"]["relation"]) == 1
    assert all(item["synthetic_data"] for item in models["nlp_checkpoints"])

    stored = client.post("/documents", json={
        "document_id": "doc-demo",
        "text": "Alice works at Acme.",
    })
    assert stored.status_code == 201
    extracted = client.post("/documents/extract", json={
        "document_id": "doc-demo",
        "ner_checkpoint": models["nlp_by_task"]["ner"][0],
        "relation_checkpoint": models["nlp_by_task"]["relation"][0],
    })
    assert extracted.status_code == 200
    assert extracted.json["inserted_into_graph"] is False
    assert extracted.json["candidate_relations"]
    candidate_index = next(
        index for index, candidate in enumerate(extracted.json["candidate_relations"])
        if candidate["subject"] == "Alice" and candidate["label"] == "works_at"
    )
    candidate = extracted.json["candidate_relations"][candidate_index]
    assert candidate["synthetic_model"] is True
    assert candidate["document_id"] == "doc-demo"
    assert candidate["evidence_snippet"] == "Alice works at Acme."

    approved = client.post(
        "/documents/doc-demo/review",
        json={"candidate_index": candidate_index, "decision": "approve"},
    )
    assert approved.status_code == 200
    assert approved.json["inserted"] is True
    relation = next(item for item in client.get("/graph").json["relations"] if item["subject"] == "Alice")
    assert relation["provenance"]["document_id"] == "doc-demo"
    graph = KnowledgeGraph.load_json(tmp_path / "graph.json")
    assert graph.query(subject="Alice")[0].provenance["document_id"] == "doc-demo"


def test_api_persists_failed_background_training_job(tmp_path):
    from knowledge_graph_builder.app import create_app

    examples = relation_examples()
    for example in examples:
        example.relations[0]["label"] = "only_one_class"
    dataset_path = tmp_path / "single-class.json"
    dataset_path.write_text(
        json.dumps({"examples": [example.__dict__ for example in examples]}),
        encoding="utf-8",
    )
    app = create_app(
        KnowledgeGraph("api"),
        {
            "experiment_dir": str(tmp_path / "experiments"),
            "nlp_checkpoint_dir": str(tmp_path / "nlp-checkpoints"),
            "dataset_path": str(dataset_path),
        },
    )
    client = app.test_client()
    response = client.post("/nlp/train", json={"task": "relation"})
    assert response.status_code == 202
    job_id = response.json["job_id"]
    deadline = time.monotonic() + 10
    status = "queued"
    while status in {"queued", "running"} and time.monotonic() < deadline:
        status_response = client.get(f"/jobs/{job_id}")
        status = status_response.json["status"]
        if status in {"queued", "running"}:
            time.sleep(0.02)
    assert status == "failed"
    assert "two distinct" in client.get(f"/jobs/{job_id}").json["error"]
    assert (tmp_path / "experiments" / f"{job_id}.json").exists()
    retry = client.post(f"/jobs/{job_id}/retry")
    assert retry.status_code == 202
    retry_id = retry.json["job_id"]
    deadline = time.monotonic() + 10
    retry_status = "queued"
    while retry_status in {"queued", "running"} and time.monotonic() < deadline:
        retry_status = client.get(f"/jobs/{retry_id}").json["status"]
        if retry_status in {"queued", "running"}:
            time.sleep(0.02)
    assert retry_status == "failed"
    assert client.get(f"/jobs/{retry_id}").json["retry_of"] == job_id


def test_api_runs_real_background_training_and_reports_progress(tmp_path):
    from knowledge_graph_builder.app import create_app

    sample_dataset = Path(__file__).parent.parent / "data" / "sample_nlp_dataset.json"
    app = create_app(
        KnowledgeGraph("api"),
        {
            "experiment_dir": str(tmp_path / "experiments"),
            "nlp_checkpoint_dir": str(tmp_path / "nlp-checkpoints"),
            "dataset_path": str(sample_dataset),
        },
    )
    client = app.test_client()
    response = client.post("/nlp/train", json={"task": "relation", "config": {"epochs": 1}})
    assert response.status_code == 202
    job_id = response.json["job_id"]
    deadline = time.monotonic() + 15
    job = client.get(f"/jobs/{job_id}").json
    while job["status"] in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(0.02)
        job = client.get(f"/jobs/{job_id}").json
    assert job["status"] == "completed"
    assert job["result"]["synthetic_data"] is True
    assert "not meaningful" in job["result"]["interpretation"]
    assert any(log.startswith("epoch=1/1 train_loss=") for log in job["logs"])
    assert len(client.get("/experiments").json["experiments"]) == 1
