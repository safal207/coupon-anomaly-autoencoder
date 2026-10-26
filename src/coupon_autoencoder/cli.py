import argparse
import csv
import json
from pathlib import Path

from .data import read_features
from .model import score_bundle


def main():
    parser = argparse.ArgumentParser(description="Synthetic coupon-terminal autoencoder benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Generate, train, calibrate and evaluate")
    run_parser.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    run_parser.add_argument("--seed", type=int, default=42)
    score_parser = commands.add_parser("score", help="Score feature-only CSV with a JSON model")
    score_parser.add_argument("--model", type=Path, required=True)
    score_parser.add_argument("--input", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        from .experiment import run
        report = run(args.output, args.seed)
        print(json.dumps({"data_source": report["data_source"], "models": report["models"], "output": str(args.output)}, ensure_ascii=False, indent=2))
    else:
        if args.output.resolve() in {args.model.resolve(), args.input.resolve()}:
            parser.error("Output must not overwrite model or input")
        bundle = json.loads(args.model.read_text(encoding="utf-8"))
        scores, flags = score_bundle(bundle, read_features(args.input))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n"); w.writerow(["row_index", "reconstruction_mse", "review_flag"])
            w.writerows((i, float(s), bool(flag)) for i, (s, flag) in enumerate(zip(scores, flags)))
        print(f"Scored {len(scores)} rows; {int(flags.sum())} review flags. Output: {args.output}")
