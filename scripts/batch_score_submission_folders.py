#!/usr/bin/env python
"""Score every submission folder under a root directory and write one CSV table.

Expected folder layout:

root_dir/
  audio_prompt/
    tfq.jsonl
    mcq.jsonl
    typea_oeq.jsonl
    typeb_oeq.jsonl
  audio_postprocess/
    tfq.jsonl
    mcq.jsonl
    typea_oeq.jsonl
    typeb_oeq.jsonl
  ...

The script calls an existing TRIDENT audio scorer once per folder, reads each
score_summary.json, and writes a compact table with these columns:
Folder, TFQ_Acc, MCQ_Score, TypeB_AccDet, TypeB_Cover, TypeB_CHAIR,
TypeB_F0.5, TypeA_Cover, TypeA_CHAIR, TypeA_F0.5, TCS.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REQUIRED_FILES = ["tfq.jsonl", "mcq.jsonl", "typea_oeq.jsonl", "typeb_oeq.jsonl"]
OUTPUT_COLUMNS = [
    "Folder",
    "TFQ_Acc",
    "MCQ_Score",
    "TypeB_AccDet",
    "TypeB_Cover",
    "TypeB_CHAIR",
    "TypeB_F0.5",
    "TypeA_Cover",
    "TypeA_CHAIR",
    "TypeA_F0.5",
    "TCS",
]


def has_required_outputs(folder: Path) -> bool:
    return folder.is_dir() and all((folder / name).exists() for name in REQUIRED_FILES)


def discover_folders(root_dir: Path, recursive: bool = False) -> List[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root_dir}")

    candidates: Iterable[Path]
    if recursive:
        candidates = (path for path in root_dir.rglob("*") if path.is_dir())
    else:
        candidates = (path for path in root_dir.iterdir() if path.is_dir())

    folders = [path for path in candidates if has_required_outputs(path)]
    return sorted(folders, key=lambda p: str(p).lower())


def safe_name_for_path(path: Path, root_dir: Path) -> str:
    try:
        rel = path.relative_to(root_dir)
    except ValueError:
        rel = path.name
    text = str(rel).replace("\\", "__").replace("/", "__")
    keep = []
    for char in text:
        keep.append(char if char.isalnum() or char in {"_", "-", "."} else "_")
    return "".join(keep).strip("_") or path.name


def run_scorer(
    *,
    scorer_script: Path,
    data_root: Path,
    submission_dir: Path,
    split: str,
    modality: str,
    output_json: Path,
    oeq_parser: str,
    qwen_model_path: Optional[Path],
    parser_cache: Optional[Path],
    parser_batch_size: int,
    parser_max_new_tokens: int,
    parser_device: str,
    parser_dtype: str,
    strict: str,
    extra_args: List[str],
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(scorer_script),
        "--data-root",
        str(data_root),
        "--submission-dir",
        str(submission_dir),
        "--split",
        split,
        "--modality",
        modality,
        "--output-json",
        str(output_json),
        "--oeq-parser",
        oeq_parser,
        "--strict",
        strict,
    ]

    if oeq_parser == "qwen":
        if qwen_model_path is None:
            raise ValueError("--qwen-model-path is required when --oeq-parser qwen")
        cmd.extend(["--qwen-model-path", str(qwen_model_path)])
        if parser_cache is not None:
            cmd.extend(["--parser-cache", str(parser_cache)])
        cmd.extend(
            [
                "--parser-batch-size",
                str(parser_batch_size),
                "--parser-max-new-tokens",
                str(parser_max_new_tokens),
                "--parser-device",
                parser_device,
                "--parser-dtype",
                parser_dtype,
            ]
        )

    cmd.extend(extra_args)
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def get_nested(summary: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    value: Any = summary
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def flatten_summary(folder: Path, root_dir: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    try:
        folder_name = str(folder.relative_to(root_dir))
    except ValueError:
        folder_name = folder.name
    return {
        "Folder": folder_name.replace("\\", "/"),
        "TFQ_Acc": get_nested(summary, "tfq", "acc_tfq"),
        "MCQ_Score": get_nested(summary, "mcq", "score_mcq"),
        "TypeB_AccDet": get_nested(summary, "typeb_oeq", "acc_det"),
        "TypeB_Cover": get_nested(summary, "typeb_oeq", "cover"),
        "TypeB_CHAIR": get_nested(summary, "typeb_oeq", "chair"),
        "TypeB_F0.5": get_nested(summary, "typeb_oeq", "f_0_5"),
        "TypeA_Cover": get_nested(summary, "typea_oeq", "cover"),
        "TypeA_CHAIR": get_nested(summary, "typea_oeq", "chair"),
        "TypeA_F0.5": get_nested(summary, "typea_oeq", "f_0_5"),
        "TCS": get_nested(summary, "tcs", "TCS"),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No score rows to display.")
        return
    widths = {col: len(col) for col in OUTPUT_COLUMNS}
    formatted_rows: List[Dict[str, str]] = []
    for row in rows:
        formatted: Dict[str, str] = {}
        for col in OUTPUT_COLUMNS:
            value = row.get(col, "")
            if isinstance(value, float):
                text = f"{value:.6f}"
            else:
                text = str(value)
            formatted[col] = text
            widths[col] = max(widths[col], len(text))
        formatted_rows.append(formatted)

    header = "  ".join(col.ljust(widths[col]) for col in OUTPUT_COLUMNS)
    sep = "  ".join("-" * widths[col] for col in OUTPUT_COLUMNS)
    print(header)
    print(sep)
    for row in formatted_rows:
        print("  ".join(row[col].ljust(widths[col]) for col in OUTPUT_COLUMNS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True, help="Directory containing submission subfolders to score.")
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/trident"))
    parser.add_argument("--scorer-script", type=Path, default=Path("score_audio_submission.py"), help="Path to score_audio_submission.py.")
    parser.add_argument("--split", default="public_val")
    parser.add_argument("--modality", default="audio")
    parser.add_argument("--output-csv", type=Path, default=None, help="CSV summary path. Defaults to <root-dir>/batch_scorer_summary.csv.")
    parser.add_argument("--score-json-dir", type=Path, default=None, help="Directory for per-folder score_summary.json files. Defaults to <root-dir>/_score_json.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for folders that contain all required JSONL files.")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing per-folder score JSON files when present.")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sort-by", default="TCS", choices=OUTPUT_COLUMNS, help="Column to sort by. Use Folder for alphabetical order.")
    parser.add_argument("--ascending", action="store_true", help="Sort ascending instead of descending.")
    parser.add_argument("--oeq-parser", choices=("regex", "qwen"), default="qwen")
    parser.add_argument("--qwen-model-path", type=Path, default=Path("/root/Qwen3.5-4B"))
    parser.add_argument("--parser-cache-dir", type=Path, default=None, help="Directory for Qwen OEQ parser caches. Defaults to <root-dir>/_parser_cache.")
    parser.add_argument("--parser-batch-size", type=int, default=32)
    parser.add_argument("--parser-max-new-tokens", type=int, default=128)
    parser.add_argument("--parser-device", default="cuda")
    parser.add_argument("--parser-dtype", default="bfloat16")
    parser.add_argument("--strict", default="true")
    parser.add_argument("--extra-scorer-args", nargs=argparse.REMAINDER, default=[], help="Additional args passed to score_audio_submission.py after --.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = args.root_dir.resolve()
    scorer_script = args.scorer_script.resolve()
    if not scorer_script.exists():
        raise FileNotFoundError(f"Scorer script not found: {scorer_script}")

    output_csv = args.output_csv or (root_dir / "batch_scorer_summary.csv")
    score_json_dir = args.score_json_dir or (root_dir / "_score_json")
    parser_cache_dir = args.parser_cache_dir or (root_dir / "_parser_cache")
    score_json_dir.mkdir(parents=True, exist_ok=True)
    if args.oeq_parser == "qwen":
        parser_cache_dir.mkdir(parents=True, exist_ok=True)

    folders = discover_folders(root_dir, recursive=args.recursive)
    if not folders:
        raise RuntimeError(f"No valid submission folders found under {root_dir}. Each folder must contain: {', '.join(REQUIRED_FILES)}")

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for index, folder in enumerate(folders, start=1):
        print(f"\n[{index}/{len(folders)}] Scoring folder: {folder}")
        safe_name = safe_name_for_path(folder, root_dir)
        output_json = score_json_dir / f"{safe_name}.score_summary.json"
        parser_cache = parser_cache_dir / f"{safe_name}.oeq_parser_cache.jsonl" if args.oeq_parser == "qwen" else None

        try:
            if not (args.reuse_existing and output_json.exists()):
                result = run_scorer(
                    scorer_script=scorer_script,
                    data_root=args.data_root,
                    submission_dir=folder,
                    split=args.split,
                    modality=args.modality,
                    output_json=output_json,
                    oeq_parser=args.oeq_parser,
                    qwen_model_path=args.qwen_model_path,
                    parser_cache=parser_cache,
                    parser_batch_size=args.parser_batch_size,
                    parser_max_new_tokens=args.parser_max_new_tokens,
                    parser_device=args.parser_device,
                    parser_dtype=args.parser_dtype,
                    strict=args.strict,
                    extra_args=list(args.extra_scorer_args or []),
                )
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                if result.returncode != 0:
                    raise RuntimeError(f"Scorer exited with code {result.returncode}")

            summary = read_json(output_json)
            rows.append(flatten_summary(folder, root_dir, summary))
        except Exception as exc:
            message = str(exc)
            errors.append({"Folder": str(folder), "Error": message})
            print(f"ERROR scoring {folder}: {message}", file=sys.stderr)
            if not args.continue_on_error:
                raise

    if args.sort_by == "Folder":
        rows.sort(key=lambda row: str(row.get("Folder", "")), reverse=not args.ascending)
    else:
        rows.sort(key=lambda row: float(row.get(args.sort_by, 0.0)), reverse=not args.ascending)

    write_csv(output_csv, rows)
    print(f"\nWrote summary CSV: {output_csv}")
    print_table(rows)

    if errors:
        error_csv = output_csv.with_name(output_csv.stem + "_errors.csv")
        with error_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Folder", "Error"])
            writer.writeheader()
            writer.writerows(errors)
        print(f"\n{len(errors)} folder(s) failed. Error report: {error_csv}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
