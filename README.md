# KnowGraph AI

KnowGraph AI is a local knowledge-graph research workspace. It combines provenance-preserving graph storage, trainable NLP and GNN models, evidence-grounded verification and retrieval, a Flask API, and a responsive web interface.

The application does not bundle benchmark corpora or claim benchmark performance. Pretrained transformer runs require a locally available model or a download from Hugging Face. Small synthetic examples are for pipeline smoke tests only.

## What is implemented

- **NER:** Hugging Face token classification fine-tuning, word-to-subword label alignment, and a separate lightweight, trainable local PyTorch token classifier for offline workflow tests. The local model is trained from scratch and is not a pretrained transformer.
- **Relation classification:** trainable PyTorch text/span classifier; each example defines one relation label and subject/object character spans. Evaluation includes a training-split majority-class baseline. This is not a document-level DocRED model.
- **Link prediction:** PyTorch Geometric R-GCN encoder with DistMult scoring, negative sampling, validation loss, filtered head-and-tail MRR/Hits@K, and a train-graph degree baseline. Fixed WN18RR/FB15k-237 split loading is supported. Predicted edges are candidates, never facts.
- **Claim verification:** `SUPPORTED`, `REFUTED`, and `NOT ENOUGH INFORMATION` are distinct. Missing graph facts do not imply refutation. Optional transformer NLI fine-tuning reports accuracy, macro-F1, per-class results, and confusion matrix.
- **FEVER preparation/evaluation:** the local JSONL adapter preserves claim labels and evidence page/sentence references. Evidence retrieval is evaluated separately from classification against source/document IDs already present in the graph; no passages or citations are fabricated.
- **GraphRAG:** retrieves source-backed graph facts and provenance. Predictions are omitted from evidence by default.
- **Model Lab:** background NLP/GNN/NLI jobs, validation, status/log polling, losses, checkpoint discovery, and inference endpoints.
- **Web application:** Overview, Knowledge Graph, Documents, Claim Verification, GraphRAG Assistant, Evidence Explorer, Model Lab, Dataset Manager, Experiments, and Settings. Counts and activity come from the API and local stores; no trend charts or decorative performance claims are generated.

## Install and run

Python 3.11–3.14 is supported by this source tree. Install a PyTorch wheel appropriate for the host first, then the remaining requirements.

### CPU

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r requirements.txt
```

### Optional GPU

Install PyTorch using the official selector for the host CUDA version, then install `requirements.txt`. PyTorch Geometric uses that PyTorch installation. `device=auto` uses CUDA when available; `cpu` forces CPU. Verify CUDA availability in the chosen environment before starting large experiments.

Start the web workspace:

```bash
python -m knowledge_graph_builder
```

Open `http://127.0.0.1:5000`. The app binds to loopback by default. The vanilla HTML/CSS/JavaScript frontend is served by Flask; there is no separate Node build or external font/runtime dependency.

The default graph is persisted at `data/knowledge_graph.json` and source documents at `data/documents.json`. Configure `graph_path`, `documents_path`, `dataset_dir`, and checkpoint/experiment roots through `create_app(..., model_config={...})` when embedding the app. Do not expose this development API publicly without authentication and deployment hardening.

The default NER model is `distilbert/distilbert-base-uncased`; NLI defaults to `cross-encoder/nli-distilroberta-base`. First use may download model files (hundreds of MB, depending on model). For offline training, use `--pretrained-model /local/model/directory`. If a model cannot be loaded, training fails explicitly; no untrained transformer is presented as fine-tuned.

### Offline document-extraction setup

The Documents page requires a compatible saved NER checkpoint and relation checkpoint. If either is missing, use **Train lightweight demo checkpoints**. This starts real background PyTorch training for both tasks on `data/sample_nlp_dataset.json`; the saved metadata marks these checkpoints as synthetic smoke models. They let you exercise extraction, candidate review, and provenance locally, but are not evidence of useful accuracy and should not be used to extract research facts without careful review.

For a local NER run on your own labeled dataset, select **Lightweight local PyTorch** in Model Lab. It uses aligned `ner_tokens` and `ner_tags`, trains a local feature/embedding classifier from scratch, saves weights and label/feature maps under `checkpoints/nlp/ner-local-*`, and is available without downloading a pretrained model. To avoid the open-vocabulary false positives common to a tiny from-scratch model, unseen tokens are conservatively labeled `O`; this makes it unsuitable for general entity extraction. For real document use, select **Pretrained transformer fine-tuning** and provide a cached model or a model ID with network access, and train on appropriately labeled data. Relation checkpoints are trained separately from the same labeled JSON format and saved under `checkpoints/nlp/relation-*`. Both model types must be saved and selected for document extraction.

The explicit offline local NER commands are:

```bash
python -m knowledge_graph_builder.cli train-nlp \
  --task ner --dataset data/sample_nlp_dataset.json \
  --ner-architecture local-context --checkpoint-dir checkpoints/nlp \
  --epochs 20 --learning-rate 0.01 --seed 42 --device cpu
python -m knowledge_graph_builder.cli eval-nlp \
  --task ner --dataset data/sample_nlp_dataset.json \
  --checkpoint checkpoints/nlp/ner-local-<run-id>
```

Replace the sample path with a prepared labeled dataset for real use. The bundled dataset is synthetic and its metrics are only smoke-test output. Evaluation must use the exact dataset fingerprint recorded by the checkpoint.

The programmatic first-run equivalent is to POST `{}` to `/nlp/demo-setup`; it returns a background job ID. Poll `/jobs/<job_id>` until completion, then retrieve the compatible paths from `/models` (`nlp_by_task.ner` and `nlp_by_task.relation`). The endpoint performs actual training and returns logs, losses, evaluation, and an explicit synthetic-data warning; it does not install pre-trained or untrained placeholder checkpoints.

An extraction candidate is a model prediction, not a verified fact. Each candidate retains its source document ID, snippet, and offsets. Only a human approval adds the relationship to the graph with provenance. Extraction is best-effort, may miss facts, and is bounded by the configured candidate-pair budget.

## Dataset access and preparation

No dataset downloader is included. Obtain files from their maintained source, accept the applicable terms, and verify that the intended research/commercial use is permitted. Preserve the source release and checksum alongside experiment records. Dataset access and redistribution terms can differ by mirror.

### CoNLL-2003 NER

Use the official CoNLL-2003 distribution or an authorized mirror. Review the dataset license/access conditions before downloading or redistributing it. The parser accepts the standard sentence-separated IOB-style file with the entity tag in the last column. It retains token labels and the official split assignment; subword alignment is performed by the tokenizer during fine-tuning.

```bash
python -m knowledge_graph_builder.cli prepare-conll \
  --input /datasets/conll/train.txt --split train --output data/conll-train.json
python -m knowledge_graph_builder.cli prepare-conll \
  --input /datasets/conll/valid.txt --split validation --output data/conll-validation.json
python -m knowledge_graph_builder.cli prepare-conll \
  --input /datasets/conll/test.txt --split test --output data/conll-test.json
python -m knowledge_graph_builder.cli combine-datasets \
  --train data/conll-train.json --validation data/conll-validation.json \
  --test data/conll-test.json --name CoNLL-2003 --output data/conll2003.json
```

Run actual pretrained fine-tuning and held-out evaluation:

```bash
python -m knowledge_graph_builder.cli train-nlp \
  --task ner --dataset data/conll2003.json \
  --pretrained-model distilbert/distilbert-base-uncased \
  --checkpoint-dir checkpoints/nlp --epochs 3 --batch-size 8 --seed 42 --device cpu
python -m knowledge_graph_builder.cli eval-nlp \
  --task ner --dataset data/conll2003.json \
  --checkpoint checkpoints/nlp/ner-<run-id>
```

NER metrics use exact entity boundaries and labels on the held-out split. The all-outside baseline predicts no entities. Do not use the bundled synthetic sample as a benchmark.

### Relation classification / DocRED preparation

DocRED is available from the [maintained DocRED repository](https://github.com/thunlp/DocRED). Review that release's license and terms. Supply `rel_info.json` so IDs such as `P26` map explicitly to relation names; the converter retains each relation's evidence sentence IDs and passages and uses the first mention of each entity in the full document text. Each output row represents one subject/object/relation annotation. The default `--negative-ratio 1` adds a deterministic sample of unannotated ordered entity pairs labeled `NO_RELATION`; this is an explicit preparation choice, not the official DocRED metric.

```bash
python -m knowledge_graph_builder.cli prepare-docred \
  --input /datasets/DocRED/train_annotated.json --relation-map /datasets/DocRED/rel_info.json \
  --split train --negative-ratio 1 --output data/docred-train.json
python -m knowledge_graph_builder.cli prepare-docred \
  --input /datasets/DocRED/dev.json --relation-map /datasets/DocRED/rel_info.json \
  --split validation --output data/docred-validation.json
# Provide a separately labeled test file if one is available and permitted.
python -m knowledge_graph_builder.cli prepare-docred \
  --input /datasets/DocRED/labeled-test.json --relation-map /datasets/DocRED/rel_info.json \
  --split test --output data/docred-test.json
python -m knowledge_graph_builder.cli combine-datasets \
  --train data/docred-train.json --validation data/docred-validation.json \
  --test data/docred-test.json --name DocRED --output data/docred.json
python -m knowledge_graph_builder.cli train-nlp \
  --task relation --dataset data/docred.json --checkpoint-dir checkpoints/nlp \
  --epochs 5 --batch-size 16 --seed 42 --device cpu
python -m knowledge_graph_builder.cli eval-nlp \
  --task relation --dataset data/docred.json \
  --checkpoint checkpoints/nlp/relation-<run-id>
```

The public DocRED test labels may not be available in a given release. Do not substitute unlabeled test rows or represent a random training subset as an official test result. This project's relation model is a single-relation text/span classifier: it does not reproduce DocRED's full document-level multi-label protocol or official evaluation.

### FEVER claim/evidence evaluation

Obtain FEVER from its [dataset page](https://fever.ai/dataset/fever.html) and follow its terms. The adapter retains gold page/sentence references; it does not manufacture evidence text. To evaluate, populate a graph with source passages and provenance document IDs matching the FEVER page identifiers.

```bash
python -m knowledge_graph_builder.cli prepare-fever \
  --input /datasets/FEVER/train.jsonl --output data/fever-prepared.json
python -m knowledge_graph_builder.cli eval-fever \
  --dataset data/fever-prepared.json --graph data/fever-evidence-graph.json
# Optional NLI checkpoint:
python -m knowledge_graph_builder.cli eval-fever \
  --dataset data/fever-prepared.json --graph data/fever-evidence-graph.json \
  --nli-checkpoint checkpoints/verification/claim-nli-<run-id>
```

FEVER preparation is a claim/evidence-reference adapter, not by itself a premise/hypothesis NLI training set: actual retrieved passage text is required for NLI fine-tuning. Use `train-verifier` with labeled evidence-passage/claim pairs:

```json
{
  "examples": [
    {"premise": "Source passage text.", "hypothesis": "Claim text.", "label": "SUPPORTED"},
    {"premise": "A passage that contradicts the claim.", "hypothesis": "Claim text.", "label": "REFUTED"},
    {"premise": "An actual retrieved passage that is insufficient.", "hypothesis": "Claim text.", "label": "NOT ENOUGH INFORMATION"}
  ]
}
```

```bash
python -m knowledge_graph_builder.cli train-verifier \
  --dataset data/claim-nli.json --checkpoint-dir checkpoints/verification \
  --pretrained-model cross-encoder/nli-distilroberta-base \
  --epochs 3 --batch-size 8 --seed 42 --device cpu
python -m knowledge_graph_builder.cli eval-verifier \
  --dataset data/claim-nli.json --checkpoint checkpoints/verification/claim-nli-<run-id>
```

### WN18RR and FB15k-237 link prediction

Use the original benchmark release from its maintained source (for example, [ConvE's benchmark data repository](https://github.com/TimDettmers/ConvE)) or an authorized equivalent. Review WordNet/Freebase and release-specific usage terms. Keep the original `train.txt`, `valid.txt`, and `test.txt` TSV files together. No random re-split is applied in benchmark mode, and no downloading occurs automatically.

```bash
python -m knowledge_graph_builder.cli train-gnn \
  --benchmark-dir /datasets/WN18RR --checkpoint-dir checkpoints/gnn \
  --epochs 100 --hidden-dim 64 --seed 42 --device cpu
python -m knowledge_graph_builder.cli eval-gnn \
  --benchmark-dir /datasets/WN18RR \
  --checkpoint checkpoints/gnn/rgcn-<run-id>.pt
```

Benchmark mode requires all three non-empty split files. It trains message passing on train edges only, uses distinct forward/inverse message relation IDs, retains entity/relation mappings, filters known positives from ranking candidates, and scores both head and tail queries. It compares with a degree-only baseline computed from the training graph. Results include split sizes, dataset dimensions, runtime, device, and query count. Full WN18RR/FB15k-237 evaluation can be CPU-intensive; choose hardware accordingly. A small project graph reports insufficient data instead of link-prediction metrics.

For project graphs, save a `KnowledgeGraph` to JSON and use `train-gnn --graph path/to/graph.json`. Splits are deterministic and grouped by entity pair.

## Checkpoints, datasets, and experiment metadata

- NLP transformer checkpoints: `checkpoints/nlp/<task>-<timestamp>-<id>/`
- Relation-classifier checkpoints: `checkpoints/nlp/relation-<timestamp>-<id>/`
- R-GCN checkpoints: `checkpoints/gnn/rgcn-<timestamp>-<id>.pt`
- Claim NLI checkpoints: `checkpoints/verification/claim-nli-<timestamp>-<id>/`
- Experiment/job metadata: `checkpoints/experiments/`

Each checkpoint includes weights and task mappings/configuration. Training runs use new versioned paths rather than overwriting prior checkpoints. Keep the source dataset version, access terms, and any local preprocessing notes with the experiment if reproducibility requires them.

## API and web application

`python -m knowledge_graph_builder` serves the UI and same-origin API on loopback:

- `GET /`, `GET /health`, `GET /overview`, `GET /graph`
- `POST /graph/fact`, `POST /graph/review`
- `GET /documents`, `POST /documents`, `POST /documents/upload`, `POST /documents/extract`, `POST /documents/<id>/review`
- `GET /datasets`, `POST /nlp/dataset/validate`, `POST /verify/dataset/validate`, `POST /gnn/dataset/validate`
- `POST /nlp/train`, `POST /gnn/train`, `POST /verify/train`
- `GET /jobs/<job_id>`, `POST /jobs/<job_id>/cancel`, `POST /jobs/<job_id>/retry`, `GET /experiments`, `GET /models`, `POST /models/load`
- `POST /nlp/infer`, `POST /gnn/predict`, `POST /verify`, `POST /rag`

In **Documents**, upload UTF-8 TXT/Markdown, text-based PDF, or DOCX (16 MB request limit by default), or paste text. PDF uses `pypdf`; scanned/image PDFs need OCR before upload. Select saved NER and relation checkpoints, then use **Upload + extract** or upload first and extract later. Extraction chunks long text, finds model-labeled entity spans, scores directed entity pairs, and keeps source snippets and offsets. These are best-effort model predictions—not a guarantee that every fact has been found or is true. Approve candidates individually to create graph nodes/edges with document provenance; rejected/unreviewed candidates never enter the graph.

Uploads are stored under `data/uploads/`, extracted text/document metadata under `data/documents.json`, and the default graph under `data/knowledge_graph.json`. Configure `uploads_path`, `max_upload_bytes`, and the other roots through `create_app(..., model_config={...})`. The graph explorer uses real graph entities/relationships with selection, search, relation filters, wheel zoom, and drag pan. The assistant cites only source IDs returned by graph retrieval.

## Tests and honest reporting

```bash
python -m pytest -q
python -m compileall -q knowledge_graph_builder tests
node --check knowledge_graph_builder/static/app.js
```

Fast tests cover actual parameter updates, checkpoint reload/inference, deterministic and fixed dataset splits, DocRED/CoNLL/FEVER preparation, GNN leakage and negative sampling, head/tail ranking metrics, API validation/background jobs, graph persistence, evidence provenance, and frontend serving. Transformer-download integration tests are not run by the default suite; the local tiny-config tests exercise actual model optimization without external downloads.

No public benchmark has been downloaded or trained in this repository session. Local smoke experiments/tests are not meaningful model-quality estimates. Report only runs and metrics actually produced with the selected data and hardware.
