from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .gnn import GraphNeuralNetworkTrainer
from .graph import KnowledgeGraph
from .nlp import NLPTrainingPipeline, dataset_digest, split_dataset
from .dataset_prep import combine_datasets, load_kg_benchmark, prepare_conll2003, prepare_docred, prepare_fever
from .verifier import ClaimNLITrainer, ClaimVerifier


def _print(value: Dict[str, Any]) -> None:
    print(json.dumps(value, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate local NLP and graph models.")
    commands = parser.add_subparsers(dest="command", required=True)

    train_nlp = commands.add_parser("train-nlp")
    train_nlp.add_argument("--task", choices=("relation", "ner"), required=True)
    train_nlp.add_argument("--dataset", required=True)
    train_nlp.add_argument("--checkpoint-dir", default="checkpoints/nlp")
    train_nlp.add_argument("--pretrained-model", default="distilbert/distilbert-base-uncased")
    train_nlp.add_argument(
        "--ner-architecture",
        choices=("pretrained-transformer", "local-context"),
        default="pretrained-transformer",
        help="NER model architecture; local-context trains a lightweight PyTorch model from scratch.",
    )
    train_nlp.add_argument("--epochs", type=int, default=3)
    train_nlp.add_argument("--batch-size", type=int, default=8)
    train_nlp.add_argument("--learning-rate", type=float)
    train_nlp.add_argument("--seed", type=int, default=42)
    train_nlp.add_argument("--device", default="auto", help="auto, cpu, or a supported device such as cuda")

    eval_nlp = commands.add_parser("eval-nlp")
    eval_nlp.add_argument("--task", choices=("relation", "ner"), required=True)
    eval_nlp.add_argument("--dataset", required=True)
    eval_nlp.add_argument("--checkpoint", required=True)

    train_gnn = commands.add_parser("train-gnn")
    train_source = train_gnn.add_mutually_exclusive_group(required=True)
    train_source.add_argument("--graph", help="Saved KnowledgeGraph JSON.")
    train_source.add_argument("--benchmark-dir", help="Directory containing standard train.txt, valid.txt, and test.txt files.")
    train_gnn.add_argument("--checkpoint-dir", default="checkpoints/gnn")
    train_gnn.add_argument("--epochs", type=int, default=100)
    train_gnn.add_argument("--seed", type=int, default=42)
    train_gnn.add_argument("--hidden-dim", type=int, default=64)
    train_gnn.add_argument("--device", default="auto", help="auto, cpu, or a supported device such as cuda")

    eval_gnn = commands.add_parser("eval-gnn")
    eval_source = eval_gnn.add_mutually_exclusive_group(required=True)
    eval_source.add_argument("--graph", help="Saved KnowledgeGraph JSON.")
    eval_source.add_argument("--benchmark-dir", help="Directory containing standard benchmark split files.")
    eval_gnn.add_argument("--checkpoint", required=True)

    train_nli = commands.add_parser("train-verifier")
    train_nli.add_argument("--dataset", required=True)
    train_nli.add_argument("--checkpoint-dir", default="checkpoints/verification")
    train_nli.add_argument("--pretrained-model", default="cross-encoder/nli-distilroberta-base")
    train_nli.add_argument("--epochs", type=int, default=3)
    train_nli.add_argument("--batch-size", type=int, default=8)
    train_nli.add_argument("--seed", type=int, default=42)
    train_nli.add_argument("--device", default="auto")

    eval_nli = commands.add_parser("eval-verifier")
    eval_nli.add_argument("--dataset", required=True)
    eval_nli.add_argument("--checkpoint", required=True)

    prepare_conll = commands.add_parser("prepare-conll")
    prepare_conll.add_argument("--input", required=True, help="Locally obtained CoNLL-2003 IOB file.")
    prepare_conll.add_argument("--split", choices=("train", "validation", "test"), required=True)
    prepare_conll.add_argument("--output", required=True)

    prepare_docred_cmd = commands.add_parser("prepare-docred")
    prepare_docred_cmd.add_argument("--input", required=True, help="Locally obtained labeled DocRED JSON file.")
    prepare_docred_cmd.add_argument("--relation-map", required=True, help="DocRED rel_info.json.")
    prepare_docred_cmd.add_argument("--split", choices=("train", "validation", "test"), required=True)
    prepare_docred_cmd.add_argument("--output", required=True)
    prepare_docred_cmd.add_argument("--negative-ratio", type=int, default=1)

    combine = commands.add_parser("combine-datasets")
    combine.add_argument("--train", required=True)
    combine.add_argument("--validation", required=True)
    combine.add_argument("--test", required=True)
    combine.add_argument("--name", required=True)
    combine.add_argument("--output", required=True)

    prepare_fever_cmd = commands.add_parser("prepare-fever")
    prepare_fever_cmd.add_argument("--input", required=True, help="Locally obtained FEVER JSONL file.")
    prepare_fever_cmd.add_argument("--output", required=True)

    eval_fever = commands.add_parser("eval-fever")
    eval_fever.add_argument("--dataset", required=True, help="Prepared FEVER claims JSON.")
    eval_fever.add_argument("--graph", required=True, help="Graph populated with evidence and matching source/document IDs.")
    eval_fever.add_argument("--nli-checkpoint", help="Optional locally fine-tuned NLI checkpoint directory.")

    args = parser.parse_args()
    if args.command == "train-nlp":
        config = {
            "dataset_path": args.dataset,
            "checkpoint_dir": args.checkpoint_dir,
            "pretrained_model": args.pretrained_model,
            "ner_architecture": args.ner_architecture,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "random_seed": args.seed,
            "device": args.device,
        }
        if args.learning_rate is not None:
            config["learning_rate"] = args.learning_rate
        pipeline = NLPTrainingPipeline(config)
        examples = pipeline.load_dataset()
        _print(pipeline.train(examples, task=args.task))
    elif args.command == "eval-nlp":
        pipeline = NLPTrainingPipeline({"dataset_path": args.dataset})
        examples = pipeline.load_dataset()
        if args.task == "relation":
            result = pipeline.evaluate_relation_checkpoint(args.checkpoint, examples)
        else:
            metadata = json.loads((Path(args.checkpoint) / "metadata.json").read_text(encoding="utf-8"))
            selected = [example for example in examples if example.ner_tokens]
            if dataset_digest(selected) != metadata["dataset_sha256"]:
                raise ValueError("Evaluation dataset does not match the checkpoint's recorded dataset fingerprint.")
            test = split_dataset(selected, int(metadata["seed"]))["test"]
            result = pipeline.evaluate_ner(test, args.checkpoint)
        _print(result)
    elif args.command == "train-gnn":
        if args.benchmark_dir:
            graph, benchmark_splits = load_kg_benchmark(args.benchmark_dir)
            benchmark_splits = {
                name: [list(triple) for triple in rows]
                for name, rows in benchmark_splits.items()
            }
        else:
            graph = KnowledgeGraph.load_json(args.graph)
            benchmark_splits = None
        trainer = GraphNeuralNetworkTrainer(
            graph,
            {
                "checkpoint_dir": args.checkpoint_dir,
                "epochs": args.epochs,
                "random_seed": args.seed,
                "hidden_dim": args.hidden_dim,
                "device": args.device,
                "benchmark_splits": benchmark_splits,
                "dataset_name": Path(args.benchmark_dir).name if args.benchmark_dir else Path(args.graph).stem,
            },
        )
        _print(trainer.train())
    elif args.command == "eval-gnn":
        graph = load_kg_benchmark(args.benchmark_dir)[0] if args.benchmark_dir else KnowledgeGraph.load_json(args.graph)
        trainer = GraphNeuralNetworkTrainer.load_checkpoint(args.checkpoint, graph)
        _print(trainer.evaluate_checkpoint())
    elif args.command == "train-verifier":
        trainer = ClaimNLITrainer(
            {
                "dataset_path": args.dataset,
                "checkpoint_dir": args.checkpoint_dir,
                "pretrained_model": args.pretrained_model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "random_seed": args.seed,
                "device": args.device,
            }
        )
        _print(trainer.train())
    elif args.command == "eval-verifier":
        trainer = ClaimNLITrainer()
        rows = trainer.load_dataset(args.dataset)
        metadata = json.loads((Path(args.checkpoint) / "metadata.json").read_text(encoding="utf-8"))
        if dataset_digest(rows) != metadata["dataset_sha256"]:
            raise ValueError("Evaluation dataset does not match the checkpoint's recorded dataset fingerprint.")
        trainer.seed = int(metadata["seed"])
        test = trainer._split(rows)["test"]
        _print(ClaimNLITrainer.evaluate_checkpoint(args.checkpoint, test))
    elif args.command == "prepare-conll":
        _print({"dataset": "CoNLL-2003", "split": args.split, "output": str(prepare_conll2003(args.input, args.output, args.split))})
    elif args.command == "prepare-docred":
        _print({
            "dataset": "DocRED",
            "split": args.split,
            "negative_ratio": args.negative_ratio,
            "output": str(prepare_docred(args.input, args.output, args.relation_map, args.split, args.negative_ratio)),
        })
    elif args.command == "combine-datasets":
        output = combine_datasets(
            {"train": args.train, "validation": args.validation, "test": args.test},
            args.output,
            args.name,
        )
        _print({"dataset": args.name, "output": str(output)})
    elif args.command == "prepare-fever":
        _print({"dataset": "FEVER", "output": str(prepare_fever(args.input, args.output))})
    elif args.command == "eval-fever":
        dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
        examples = dataset.get("examples") if isinstance(dataset, dict) else None
        if not isinstance(examples, list) or not examples:
            raise ValueError("Prepared FEVER file must contain a non-empty examples array.")
        graph = KnowledgeGraph.load_json(args.graph)
        config = {"nli_checkpoint": args.nli_checkpoint} if args.nli_checkpoint else {}
        _print(ClaimVerifier(graph, config).evaluate_fever(examples))


if __name__ == "__main__":
    main()
