#!/usr/bin/env python
"""Batch generator for all Qwen2.5-Omni input_source + answer_strategy combinations."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent

COMBINATIONS: List[Tuple[str, str]] = [
    ("audio", "prompt"),
    ("audio", "postprocess"),
    ("evidence", "prompt"),
    ("evidence", "postprocess"),
    ("audio_evidence", "prompt"),
    ("audio_evidence", "postprocess"),
]


def output_name(input_source: str, answer_strategy: str) -> str:
    return f"{input_source}_{answer_strategy}"


def run_one(script_path: Path, input_source: str, answer_strategy: str, base_output_dir: Path, args: argparse.Namespace) -> int:
    out_dir = base_output_dir / output_name(input_source, answer_strategy)
    cmd = [
        sys.executable,
        str(script_path),
        "--input_source", input_source,
        "--answer_strategy", answer_strategy,
        "--output_dir", str(out_dir),
    ]

    passthrough = [
        ("--split", args.split),
        ("--audio_root", args.audio_root),
        ("--question_root", args.question_root),
        ("--prob_csv", args.prob_csv),
        ("--model_path", args.model_path),
        ("--threshold_json", args.threshold_json),
        ("--prompt_template", args.prompt_template),
        ("--mode", args.mode),
        ("--observation_max_new_tokens", args.observation_max_new_tokens),
        ("--max_new_tokens", args.max_new_tokens),
        ("--tfq_max_new_tokens", args.tfq_max_new_tokens),
        ("--mcq_max_new_tokens", args.mcq_max_new_tokens),
        ("--typea_max_new_tokens", args.typea_max_new_tokens),
        ("--typeb_max_new_tokens", args.typeb_max_new_tokens),
        ("--temperature", args.temperature),
        ("--top_p", args.top_p),
        ("--num_samples", args.num_samples),
        ("--first_n_questions", args.first_n_questions),
        ("--batch_size", args.batch_size),
        ("--device", args.device),
        ("--dtype", args.dtype),
        ("--prompt_examples_per_task", args.prompt_examples_per_task),
    ]
    for flag, value in passthrough:
        if value is not None:
            cmd.extend([flag, str(value)])

    if args.rule_fallback:
        cmd.append("--rule_fallback")
    if args.disable_llm:
        cmd.append("--disable_llm")
    if args.dry_run:
        cmd.append("--dry_run")
    if args.allow_missing:
        cmd.append("--allow_missing")
    if args.save_prompt_examples:
        cmd.append("--save_prompt_examples")

    print(f"\n{'=' * 70}")
    print(f"Running Qwen2.5-Omni: {input_source}_{answer_strategy}")
    print(f"Output dir: {out_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 70}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_output_dir", type=Path, default=Path("/root/project/audio-deepfake-main/audio-deepfake-main/outputs/batch_all_combinations_qwen25_omni"))
    p.add_argument("--script_path", type=Path, default=SCRIPT_DIR / "generate_all_audio_answers_qwen25_omni_refactored.py")
    p.add_argument("--split", default="public_val", choices=["train", "public_val", "private_test"])
    p.add_argument("--audio_root", type=Path, default=None)
    p.add_argument("--question_root", type=Path, default=Path("/root/autodl-tmp/trident"))
    p.add_argument("--prob_csv", type=Path, default=Path("/root/project/audio-deepfake-main/audio-deepfake-main/evidence_files/val_predictions_epoch_15.csv"))
    p.add_argument("--model_path", default="/root/autodl-tmp/Qwen2.5-Omni-3B")
    p.add_argument("--threshold_json", type=Path, default=Path("/root/project/audio-deepfake-main/audio-deepfake-main/evidence_files/audio_thresholds.json"))
    p.add_argument("--prompt_template", type=Path, default=Path("/root/project/audio-deepfake-main/audio-deepfake-main/prompts/audio_prompt_templates.yaml"))
    p.add_argument("--mode", default="all", choices=["all", "tfq", "mcq", "typea_oeq", "typeb_oeq"])
    p.add_argument("--observation_max_new_tokens", type=int, default=192)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--tfq_max_new_tokens", type=int, default=8)
    p.add_argument("--mcq_max_new_tokens", type=int, default=16)
    p.add_argument("--typea_max_new_tokens", type=int, default=64)
    p.add_argument("--typeb_max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--rule_fallback", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--disable_llm", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--first_n_questions", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--allow_missing", action="store_true")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32", "auto"])
    p.add_argument("--save_prompt_examples", action="store_true")
    p.add_argument("--prompt_examples_per_task", type=int, default=2)
    p.add_argument(
        "--skip_combinations",
        nargs="+",
        default=[],
        choices=["audio_prompt", "audio_postprocess", "evidence_prompt", "evidence_postprocess", "audio_evidence_prompt", "audio_evidence_postprocess"],
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.script_path.exists():
        print(f"Error: could not find script at {args.script_path}", file=sys.stderr)
        return 1
    args.base_output_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for input_source, answer_strategy in COMBINATIONS:
        name = output_name(input_source, answer_strategy)
        if name in args.skip_combinations:
            print(f"\nSkipping: {name}")
            continue
        code = run_one(args.script_path, input_source, answer_strategy, args.base_output_dir, args)
        if code != 0:
            print(f"\nFAILED: {name} (exit code: {code})")
            failed.append(name)
        else:
            print(f"\nSUCCESS: {name}")
    print(f"\n{'=' * 70}")
    print("Qwen2.5-Omni batch processing complete!")
    print(f"Total combinations: {len(COMBINATIONS)}")
    print(f"Skipped: {len(args.skip_combinations)}")
    print(f"Successful: {len(COMBINATIONS) - len(failed) - len(args.skip_combinations)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed combinations: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
