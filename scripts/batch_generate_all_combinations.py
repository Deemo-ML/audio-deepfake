#!/usr/bin/env python
"""Batch generator for all input_source + answer_strategy combinations."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

# All combinations to run
COMBINATIONS: List[Tuple[str, str]] = [
    ("audio", "prompt"),
    ("audio", "postprocess"),
    ("evidence", "prompt"),
    ("evidence", "postprocess"),
    ("audio_evidence", "prompt"),
    ("audio_evidence", "postprocess"),
]


def get_output_dir_name(input_source: str, answer_strategy: str) -> str:
    """Generate output directory name from input_source and answer_strategy."""
    return f"{input_source}_{answer_strategy}"


def run_single_combination(
    script_path: Path,
    input_source: str,
    answer_strategy: str,
    base_output_dir: Path,
    args: argparse.Namespace,
) -> int:
    """Run a single combination with appropriate arguments."""
    output_dir = base_output_dir / get_output_dir_name(input_source, answer_strategy)
    cmd_args = [
        sys.executable,
        str(script_path),
        "--input_source",
        input_source,
        "--answer_strategy",
        answer_strategy,
        "--output_dir",
        str(output_dir),
    ]

    if args.split:
        cmd_args.extend(["--split", args.split])
    if args.audio_root:
        cmd_args.extend(["--audio_root", str(args.audio_root)])
    if args.question_root:
        cmd_args.extend(["--question_root", str(args.question_root)])
    if args.prob_csv:
        cmd_args.extend(["--prob_csv", str(args.prob_csv)])
    if args.model_path:
        cmd_args.extend(["--model_path", args.model_path])
    if args.threshold_json:
        cmd_args.extend(["--threshold_json", str(args.threshold_json)])
    if args.prompt_template:
        cmd_args.extend(["--prompt_template", str(args.prompt_template)])
    if args.mode:
        cmd_args.extend(["--mode", args.mode])
    if args.observation_max_new_tokens:
        cmd_args.extend(["--observation_max_new_tokens", str(args.observation_max_new_tokens)])
    if args.max_new_tokens:
        cmd_args.extend(["--max_new_tokens", str(args.max_new_tokens)])
    if args.tfq_max_new_tokens:
        cmd_args.extend(["--tfq_max_new_tokens", str(args.tfq_max_new_tokens)])
    if args.mcq_max_new_tokens:
        cmd_args.extend(["--mcq_max_new_tokens", str(args.mcq_max_new_tokens)])
    if args.typea_max_new_tokens:
        cmd_args.extend(["--typea_max_new_tokens", str(args.typea_max_new_tokens)])
    if args.typeb_max_new_tokens:
        cmd_args.extend(["--typeb_max_new_tokens", str(args.typeb_max_new_tokens)])
    if args.temperature:
        cmd_args.extend(["--temperature", str(args.temperature)])
    if args.top_p:
        cmd_args.extend(["--top_p", str(args.top_p)])
    if args.rule_fallback:
        cmd_args.append("--rule_fallback")
    if args.disable_llm:
        cmd_args.append("--disable_llm")
    if args.dry_run:
        cmd_args.append("--dry_run")
    if args.num_samples:
        cmd_args.extend(["--num_samples", str(args.num_samples)])
    if args.first_n_questions:
        cmd_args.extend(["--first_n_questions", str(args.first_n_questions)])
    if args.batch_size:
        cmd_args.extend(["--batch_size", str(args.batch_size)])
    if args.allow_missing:
        cmd_args.append("--allow_missing")
    if args.device:
        cmd_args.extend(["--device", args.device])
    if args.dtype:
        cmd_args.extend(["--dtype", args.dtype])
    if args.save_prompt_examples:
        cmd_args.append("--save_prompt_examples")
    if args.prompt_examples_per_task:
        cmd_args.extend(["--prompt_examples_per_task", str(args.prompt_examples_per_task)])

    print(f"\n{'='*70}")
    print(f"Running: {input_source}_{answer_strategy}")
    print(f"Output dir: {output_dir}")
    print(f"Command: {' '.join(cmd_args)}")
    print(f"{'='*70}")

    result = subprocess.run(cmd_args, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    
    # Output control
    p.add_argument("--base_output_dir", type=Path, 
                   default=Path("/root/project/audio-deepfake-main/audio-deepfake-main/outputs/batch_all_combinations"))
    
    # Arguments passed to the original script
    p.add_argument("--split", default="public_val", choices=["train", "public_val", "private_test"])
    p.add_argument("--audio_root", type=Path, default=None)
    p.add_argument("--question_root", type=Path, default=r"/root/autodl-tmp/trident")
    p.add_argument("--prob_csv", type=Path, default=r"/root/project/audio-deepfake-main/audio-deepfake-main/evidence_files/val_predictions_epoch_15.csv")
    p.add_argument("--model_path", default=r"/root/autodl-tmp/Qwen2-Audio-7B")
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
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--allow_missing", action="store_true")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32", "auto"])
    p.add_argument("--save_prompt_examples", action="store_true")
    p.add_argument("--prompt_examples_per_task", type=int, default=2)
    
    # Batch-specific options
    p.add_argument("--skip_combinations", nargs="+", default=[],
                   choices=["audio_prompt", "audio_postprocess", "evidence_prompt", "evidence_postprocess", 
                            "audio_evidence_prompt", "audio_evidence_postprocess"],
                   help="Combinations to skip")
    
    return p.parse_args()


def main() -> int:
    args = parse_args()
    script_path = SCRIPT_DIR / "generate_all_audio_answers_qwen2_audio_refactored.py"
    
    if not script_path.exists():
        print(f"Error: Could not find script at {script_path}", file=sys.stderr)
        return 1
    
    args.base_output_dir.mkdir(parents=True, exist_ok=True)
    
    failed_combinations = []
    
    for input_source, answer_strategy in COMBINATIONS:
        combo_name = f"{input_source}_{answer_strategy}"
        
        if combo_name in args.skip_combinations:
            print(f"\nSkipping: {combo_name}")
            continue
        
        return_code = run_single_combination(script_path, input_source, answer_strategy, args.base_output_dir, args)
        
        if return_code != 0:
            print(f"\nFAILED: {combo_name} (exit code: {return_code})")
            failed_combinations.append(combo_name)
        else:
            print(f"\nSUCCESS: {combo_name}")
    
    print(f"\n{'='*70}")
    print("Batch processing complete!")
    print(f"Total combinations: {len(COMBINATIONS)}")
    print(f"Skipped: {len(args.skip_combinations)}")
    print(f"Successful: {len(COMBINATIONS) - len(failed_combinations) - len(args.skip_combinations)}")
    print(f"Failed: {len(failed_combinations)}")
    
    if failed_combinations:
        print(f"\nFailed combinations: {', '.join(failed_combinations)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())