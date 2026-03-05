from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .embeddings import ClapEmbedder
from .inference import load_artifact, predict_with_trained_model, predict_zero_shot
from .training import run_training


def _add_embedder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--clap-model-id",
        default="laion/clap-htsat-fused",
        help="Hugging Face CLAP model id.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48_000,
        help="Target sample rate for loading audio.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=10.0,
        help="Audio chunk duration used before embedding.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of chunks processed at once by CLAP.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. 'cpu' or 'cuda'. Default auto-detect.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load CLAP model files from local cache only.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emo-classifier",
        description="Train and run an audio emo song classifier.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a supervised emo classifier.")
    train.add_argument("--csv", required=True, help="CSV with audio path + label columns.")
    train.add_argument(
        "--audio-column", default="path", help="CSV column with audio file path."
    )
    train.add_argument(
        "--label-column", default="label", help="CSV column with label values."
    )
    train.add_argument(
        "--base-dir",
        default=None,
        help="Base directory to resolve relative audio paths in CSV.",
    )
    train.add_argument(
        "--model-out",
        default="artifacts/emo_classifier.joblib",
        help="Output path for trained artifact.",
    )
    train.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split fraction (set 0 to disable).",
    )
    train.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/validation split.",
    )
    train.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip unreadable files/labels instead of stopping.",
    )
    train.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples for quick experiments.",
    )
    _add_embedder_args(train)

    predict = sub.add_parser("predict", help="Predict emo probability using trained model.")
    predict.add_argument("--audio", required=True, help="Audio file to classify.")
    predict.add_argument(
        "--model",
        required=True,
        help="Path to trained artifact produced by the train command.",
    )
    predict.add_argument(
        "--threshold", type=float, default=0.5, help="Probability threshold for label output."
    )
    _add_embedder_args(predict)

    zero_shot = sub.add_parser(
        "zero-shot",
        help="Run zero-shot CLAP comparison without supervised training.",
    )
    zero_shot.add_argument("--audio", required=True, help="Audio file to classify.")
    zero_shot.add_argument(
        "--emo-prompt",
        default="an emo rock song with emotional vocals and guitar-driven arrangement",
        help="Text prompt for the positive emo class.",
    )
    zero_shot.add_argument(
        "--non-emo-prompt",
        default="a non-emo song in another style",
        help="Text prompt for the negative class.",
    )
    _add_embedder_args(zero_shot)

    return parser


def _new_embedder(args: argparse.Namespace, model_id: str | None = None) -> ClapEmbedder:
    return ClapEmbedder(
        model_id=model_id or args.clap_model_id,
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds,
        batch_size=args.batch_size,
        device=args.device,
        local_files_only=args.local_files_only,
    )


def run_train(args: argparse.Namespace) -> dict[str, object]:
    embedder = _new_embedder(args)
    result = run_training(
        csv_path=args.csv,
        model_path=args.model_out,
        embedder=embedder,
        audio_column=args.audio_column,
        label_column=args.label_column,
        base_dir=args.base_dir,
        skip_errors=args.skip_errors,
        max_samples=args.max_samples,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    return {
        "mode": "train",
        "model_path": str(result.model_path),
        "examples_used": result.examples_used,
        "skipped_examples": result.skipped_examples,
        "metrics": result.metrics,
    }


def run_predict(args: argparse.Namespace) -> dict[str, object]:
    artifact = load_artifact(args.model)
    artifact_model_id = artifact.get("clap_model_id")
    artifact_sample_rate = int(artifact.get("sample_rate", args.sample_rate))
    artifact_chunk_seconds = float(artifact.get("chunk_seconds", args.chunk_seconds))

    embedder = ClapEmbedder(
        model_id=artifact_model_id or args.clap_model_id,
        sample_rate=artifact_sample_rate,
        chunk_seconds=artifact_chunk_seconds,
        batch_size=args.batch_size,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    return predict_with_trained_model(
        audio_path=args.audio,
        artifact=artifact,
        embedder=embedder,
        threshold=args.threshold,
    )


def run_zero_shot(args: argparse.Namespace) -> dict[str, object]:
    embedder = _new_embedder(args)
    return predict_zero_shot(
        audio_path=args.audio,
        embedder=embedder,
        emo_prompt=args.emo_prompt,
        non_emo_prompt=args.non_emo_prompt,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "train":
            output = run_train(args)
        elif args.command == "predict":
            output = run_predict(args)
        elif args.command == "zero-shot":
            output = run_zero_shot(args)
        else:
            raise ValueError(f"Unsupported command: {args.command}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
