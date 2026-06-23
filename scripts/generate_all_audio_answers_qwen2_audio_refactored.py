#!/usr/bin/env python
"""Refactored Qwen2-Audio TRIDENT answer generator with explicit PromptBuilder."""
from __future__ import annotations

import argparse, json, logging, re, sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, total=None, **_): self.iterable, self.total = iterable, total
        def __iter__(self): return iter(self.iterable or [])
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def update(self, n=1): return None
        def set_postfix(self, *_, **__): return None

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (ROOT, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.trident_audio.prompt_builder import PromptBuilder
import generate_all_audio_answers_qwen2_audio as base


def task_max_tokens(task: str, args: argparse.Namespace) -> int:
    value = {
        "tfq": args.tfq_max_new_tokens,
        "mcq": args.mcq_max_new_tokens,
        "typea_oeq": args.typea_max_new_tokens,
        "typeb_oeq": args.typeb_max_new_tokens,
    }.get(task)
    if value is not None:
        return value
    if task == "tfq": return min(args.max_new_tokens, 8)
    if task == "mcq": return min(args.max_new_tokens, 16)
    return args.max_new_tokens


def batches(records: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(records), max(1, batch_size)):
        yield records[i:i + max(1, batch_size)]


def append_log(handle: Any, record: Dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n"); handle.flush()


def generation_required(args: argparse.Namespace) -> bool:
    return args.answer_strategy == "prompt" or (args.answer_strategy == "postprocess" and args.input_source in {"audio", "audio_evidence"})


def run_generation(inferencer: base.Qwen2AudioInferencer, args: argparse.Namespace, items: List[Dict[str, Any]], max_new_tokens: int) -> List[str]:
    system_prompt = items[0]["bundle"].system_prompt
    prompts = [item["bundle"].user_prompt for item in items]
    if args.input_source == "evidence":
        return inferencer.generate_text_batch(system_prompt, prompts, max_new_tokens)
    return inferencer.generate_batch([str(item["audio_path"]) for item in items], system_prompt, prompts, max_new_tokens)


def postprocess(task: str, question: str, options: Optional[Dict[str, str]], evidence: Any, input_source: str, raw: str):
    obs: Dict[str, Dict[str, str]] = {}
    if input_source in {"audio", "audio_evidence"}:
        obs = base.parse_audio_observation(raw)
    if input_source == "audio":
        active_artifacts = base.artifacts_from_observation(obs)
        overall = obs.get("_overall", {}).get("label", "Likely Authentic")
        active_pred_fake = overall == "Likely Manipulated" or bool(active_artifacts)
    elif input_source == "audio_evidence":
        active_artifacts, active_pred_fake = list(evidence.detected_artifacts), bool(evidence.pred_fake)
    elif input_source == "evidence":
        active_artifacts, active_pred_fake = list(evidence.detected_artifacts), bool(evidence.pred_fake)
    else:
        raise ValueError(f"Unknown input_source: {input_source}")
    used_fallback = False
    if task == "tfq":
        final, used_fallback = base.final_tfq_answer(question, active_artifacts)
    elif task == "mcq":
        final, used_fallback = base.final_mcq_answer(options or {}, active_artifacts)
    elif task == "typea_oeq":
        final = base.typea_postprocess_answer(active_artifacts, obs)
    elif task == "typeb_oeq":
        final = base.typeb_postprocess_answer(active_pred_fake, active_artifacts, obs)
    else:
        raise ValueError(f"Unknown task: {task}")
    return final, used_fallback, obs, active_artifacts, active_pred_fake


def save_prompt_example(output_dir: Path, task: str, sample_id: str, debug: Dict[str, Any], index: int) -> None:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id) or f"sample_{index}"
    path = output_dir / "prompt_examples" / f"{task}_{index:02d}_{safe_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"TEMPLATE: {debug.get('template_key', '')}\n\n")
        f.write("=== BUILD STEPS ===\n")
        for step in debug.get("steps", []):
            f.write(f"\n[{step.get('name', '')}] {step.get('detail', '')}\n{step.get('preview', '')}\n")
        f.write("\n=== SYSTEM PROMPT ===\n" + str(debug.get("system_prompt", "")) + "\n")
        f.write("\n=== USER PROMPT ===\n" + str(debug.get("user_prompt", "")) + "\n")


def process_task(task: str, files: List[Path], args: argparse.Namespace, thresholds: Dict[str, Any], audio_index: Dict[str, Path], prob_index: Dict[str, Any], inferencer: Optional[base.Qwen2AudioInferencer], builder: PromptBuilder) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in files:
        loaded = base.read_question_records(path)
        logging.info("Loaded %d records from %s", len(loaded), path)
        records.extend(loaded)
    limits = []
    if args.first_n_questions is not None: limits.append(args.first_n_questions)
    if args.dry_run: limits.append(args.num_samples)
    if limits: records = records[:min(limits)]

    missing = base.validate_or_collect_missing(records, audio_index, prob_index, task)
    if missing:
        missing_path = Path(args.output_dir) / "missing_samples.json"
        existing = json.loads(missing_path.read_text(encoding="utf-8")) if missing_path.exists() else []
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        missing_path.write_text(json.dumps(existing + missing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not args.allow_missing:
            raise RuntimeError(f"{len(missing)} {task} sample(s) are missing audio/probabilities; see {missing_path}")

    output_dir = Path(args.output_dir)
    outputs, debug_records = [], []
    task_tokens = task_max_tokens(task, args)
    prompt_example_count = 0
    live = Path(args.live_log_file).open("a", encoding="utf-8", newline="\n") if args.live_log_file else None
    try:
        with tqdm(total=len(records), desc=task, unit="sample", dynamic_ncols=True) as progress:
            for batch_i, batch in enumerate(batches(records, args.batch_size), 1):
                items, skipped = [], 0
                for record in batch:
                    sid = base.get_sample_id(record)
                    if sid not in audio_index or sid not in prob_index:
                        skipped += 1; continue
                    evidence = prob_index[sid]
                    question = base.get_task_question_text(record, task)
                    options = base.parse_mcq_options(record) if task == "mcq" else None
                    bundle = builder.build(task=task, input_source=args.input_source, answer_strategy=args.answer_strategy, question=question, evidence=evidence, thresholds=thresholds, options=options)
                    prompt_debug = bundle.as_debug_dict()
                    prompt_example_count += 1
                    if args.save_prompt_examples and prompt_example_count <= args.prompt_examples_per_task:
                        save_prompt_example(output_dir, task, sid, prompt_debug, prompt_example_count)
                    items.append({"record": record, "sample_id": sid, "audio_path": audio_index[sid], "evidence": evidence, "question": question, "options": options, "bundle": bundle, "prompt_debug": prompt_debug})

                raw_outputs = [""] * len(items)
                if generation_required(args) and inferencer is not None and items:
                    max_tokens = args.observation_max_new_tokens if args.answer_strategy == "postprocess" else task_tokens
                    try:
                        raw_outputs = run_generation(inferencer, args, items, max_tokens)
                    except Exception as exc:
                        if not args.rule_fallback: raise
                        logging.warning("Generation failed for %s batch %d: %s", task, batch_i, exc)
                elif generation_required(args) and inferencer is None:
                    logging.warning("Generation required for %s but LLM is disabled", task)

                for item, raw in zip(items, raw_outputs):
                    evidence, options, question = item["evidence"], item["options"], item["question"]
                    observation_text, parsed_observation, active_artifacts, active_pred_fake = "", {}, [], bool(evidence.pred_fake)
                    if args.answer_strategy == "prompt":
                        final, used_fallback = base.minimal_format_fix(task=task, response=raw, options=options, evidence=evidence, input_source=args.input_source)
                    else:
                        observation_text = raw if args.input_source in {"audio", "audio_evidence"} else ""
                        final, used_fallback, parsed_observation, active_artifacts, active_pred_fake = postprocess(task, question, options, evidence, args.input_source, observation_text)

                    out = dict(item["record"])
                    out.setdefault("id", base.get_record_id(item["record"]) or item["sample_id"])
                    out.setdefault("sample_id", item["sample_id"])
                    out[base.response_field(item["record"])] = final
                    outputs.append(out)
                    dbg = {
                        "task": task,
                        "id": base.get_record_id(item["record"]) or item["sample_id"],
                        "sample_id": item["sample_id"],
                        "audio_path": str(item["audio_path"]),
                        "question": question,
                        "prob_fake": evidence.prob_fake,
                        "fake_threshold": float(thresholds["fake_threshold"]),
                        "pred_fake": evidence.pred_fake,
                        "artifact_probs": evidence.artifact_probs,
                        "artifact_thresholds": thresholds["artifact_thresholds"],
                        "detected_artifacts": evidence.detected_artifacts,
                        "input_source": args.input_source,
                        "answer_strategy": args.answer_strategy,
                        "prompt_template_key": item["bundle"].template_key,
                        "prompt_build_steps": item["bundle"].steps,
                        "system_prompt": item["bundle"].system_prompt,
                        "prompt": item["bundle"].user_prompt,
                        "observation_text": observation_text,
                        "parsed_observation": parsed_observation,
                        "active_artifacts": active_artifacts,
                        "active_pred_fake": active_pred_fake,
                        "raw_model_response": raw,
                        "final_response": final,
                        "used_fallback": used_fallback,
                    }
                    debug_records.append(dbg)
                    if live:
                        append_log(live, {"timestamp": datetime.now().isoformat(timespec="seconds"), "task": task, "batch": batch_i, "sample_id": item["sample_id"], "prompt_template_key": item["bundle"].template_key, "final_response": final, "used_fallback": used_fallback})
                progress.update(len(batch)); progress.set_postfix(batch=batch_i, written=len(outputs), skipped=skipped, refresh=True)
    finally:
        if live: live.close()
    base.write_jsonl(output_dir / f"{task}.jsonl", outputs)
    base.write_jsonl(output_dir / f"debug_{task}.jsonl", debug_records)
    return outputs


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="public_val", choices=["train", "public_val", "private_test"])
    p.add_argument("--audio_root", type=Path, default=None)
    p.add_argument("--question_root", type=Path, default=r"/root/autodl-tmp/trident")
    p.add_argument("--prob_csv", type=Path, default=r"/root/project/audio-deepfake-main/audio-deepfake-main/evidence_files/val_predictions_epoch_15.csv")
    p.add_argument("--model_path", default=r"/root/autodl-tmp/Qwen2-Audio-7B-Instruct")
    p.add_argument("--output_dir", type=Path, default=r"/root/project/audio-deepfake-main/audio-deepfake-main/outputs/qwen2_audio_refactored/public_val")
    p.add_argument("--live_log_file", type=Path, default=None)
    p.add_argument("--threshold_json", type=Path, default=Path("/root/project/audio-deepfake-main/audio-deepfake-main/evidence_files/audio_thresholds.json"))
    p.add_argument("--prompt_template", type=Path, default=Path("/root/project/audio-deepfake-main/audio-deepfake-main/prompts/audio_prompt_templates.yaml"))
    p.add_argument("--input_source", default="audio", choices=["audio", "audio_evidence", "evidence"])
    p.add_argument("--answer_strategy", default="prompt", choices=["prompt", "postprocess"])
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
    p.add_argument("--check_deps", action="store_true")
    p.add_argument("--save_prompt_examples", action="store_true")
    p.add_argument("--prompt_examples_per_task", type=int, default=2)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    base.setup_logging()
    raw_argv = sys.argv[1:] if argv is None else argv
    if "--check_deps" in raw_argv:
        ok, report = base.check_qwen2_audio_dependencies()
        print("\n".join(report)); return 0 if ok else 1
    args = parse_args(argv)
    requested = base.TASKS if args.mode == "all" else [args.mode]
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if args.live_log_file is None: args.live_log_file = output_dir / "live_generation.jsonl"
    args.live_log_file.parent.mkdir(parents=True, exist_ok=True); args.live_log_file.write_text("", encoding="utf-8")
    (output_dir / "missing_samples.json").write_text("[]\n", encoding="utf-8")

    thresholds = base.load_thresholds(args.threshold_json)
    audio_root = base.resolve_audio_root(args.split, args.audio_root); args.audio_root = audio_root
    base.write_run_config(args, output_dir, thresholds, audio_root)
    builder = PromptBuilder.from_yaml(args.prompt_template)
    audio_index = base.build_audio_index(audio_root)
    prob_index = base.read_probability_csv(args.prob_csv, thresholds)
    task_files = base.discover_question_files(args.question_root, args.split)

    inferencer = None
    if not args.disable_llm:
        inferencer = base.Qwen2AudioInferencer(args.model_path, device=args.device, dtype=args.dtype, temperature=args.temperature, top_p=args.top_p)
    else:
        logging.info("LLM disabled")

    expected: Dict[str, int] = {}
    for task in requested:
        files = task_files.get(task, [])
        if not files:
            base.write_jsonl(output_dir / f"{task}.jsonl", []); base.write_jsonl(output_dir / f"debug_{task}.jsonl", []); expected[task] = 0; continue
        outputs = process_task(task, files, args, thresholds, audio_index, prob_index, inferencer, builder)
        expected[task] = len(outputs)
    for task in base.TASKS:
        if task not in requested:
            base.write_jsonl(output_dir / f"{task}.jsonl", []); base.write_jsonl(output_dir / f"debug_{task}.jsonl", []); expected[task] = 0
    base.validate_outputs(output_dir, expected, answer_strategy=args.answer_strategy)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        logging.error("%s", exc)
        raise SystemExit(1)
