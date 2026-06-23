#!/usr/bin/env python
"""Refactored Qwen2.5-Omni TRIDENT audio answer generator.

This script mirrors generate_all_audio_answers_qwen2_audio_refactored.py but
uses Qwen2.5-Omni as the audio-capable model backend. Prompt construction is
still handled by src/trident_audio/prompt_builder.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (ROOT, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.trident_audio.prompt_builder import PromptBuilder
import generate_all_audio_answers_qwen2_audio as base
import generate_all_audio_answers_qwen2_audio_refactored as q2ref


def check_qwen25_omni_dependencies() -> tuple[bool, List[str]]:
    """Check dependencies required by Qwen2.5-Omni."""
    import importlib

    results: List[str] = []
    ok = True
    for module_name in ["torch", "transformers", "accelerate", "librosa", "qwen_omni_utils"]:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            results.append(f"{module_name}: OK version={version}")
        except Exception as exc:
            ok = False
            results.append(f"{module_name}: FAIL {type(exc).__name__}: {exc}")

    try:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor  # noqa: F401
        results.append("transformers.Qwen2_5OmniForConditionalGeneration/Qwen2_5OmniProcessor: OK")
    except Exception as exc:
        ok = False
        results.append(
            "transformers.Qwen2_5OmniForConditionalGeneration/Qwen2_5OmniProcessor: "
            f"FAIL {type(exc).__name__}: {exc}"
        )
    return ok, results


class Qwen25OmniInferencer:
    """Inference wrapper exposing the same API as Qwen2AudioInferencer.

    Public methods used by the pipeline:
    - generate(audio_path, system_prompt, user_prompt, max_new_tokens)
    - generate_batch(audio_paths, system_prompt, user_prompts, max_new_tokens)
    - generate_text(system_prompt, user_prompt, max_new_tokens)
    - generate_text_batch(system_prompt, user_prompts, max_new_tokens)
    """

    def __init__(self, model_path: str, device: str = "cuda", dtype: str = "bf16", temperature: float = 0.0, top_p: float = 1.0):
        ok, report = check_qwen25_omni_dependencies()
        if not ok:
            raise RuntimeError(
                "Failed to import Qwen2.5-Omni dependencies:\n"
                + "\n".join(f"- {line}" for line in report)
                + "\nInstall a recent transformers version with Qwen2.5-Omni support and qwen-omni-utils."
            )

        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        if device == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but unavailable; falling back to CPU")
            device = "cpu"

        dtype_map: Dict[str, Any] = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
            "auto": "auto",
        }
        torch_dtype = dtype_map.get(dtype.lower())
        if torch_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype}")

        kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype, "trust_remote_code": True}
        if device == "cuda":
            kwargs["device_map"] = "auto"

        self.torch = torch
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, **kwargs)
        if device != "cuda":
            self.model.to(device)
        self.model.eval()

    def _move_inputs_to_device(self, inputs: Any) -> Any:
        if hasattr(inputs, "to"):
            return inputs.to(self.device)
        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                inputs[key] = value.to(self.device)
        return inputs

    def _generate_from_inputs(self, inputs: Any, max_new_tokens: int) -> Any:
        gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": False, "return_audio": False, "use_audio_in_video": False}
        if self.temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": self.temperature, "top_p": self.top_p})
        with self.torch.inference_mode():
            try:
                return self.model.generate(**inputs, **gen_kwargs)
            except TypeError:
                gen_kwargs.pop("return_audio", None)
                gen_kwargs.pop("use_audio_in_video", None)
                return self.model.generate(**inputs, **gen_kwargs)

    def _decode_new_tokens(self, generated: Any, inputs: Any, batch_size: int) -> List[str]:
        if isinstance(generated, tuple):
            generated = generated[0]
        if "attention_mask" in inputs:
            input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        else:
            input_lengths = [inputs["input_ids"].shape[-1]] * batch_size
        output_ids = [generated[i, int(input_lengths[i]) :] for i in range(batch_size)]
        decoded = self.processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [item.strip() for item in decoded]

    @staticmethod
    def _audio_messages(audio_path: str, system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_path)},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

    @staticmethod
    def _text_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]

    def _prepare_audio_inputs(self, messages_list: List[List[Dict[str, Any]]]) -> Any:
        from qwen_omni_utils import process_mm_info

        texts = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_list]
        audios: List[Any] = []
        images: List[Any] = []
        videos: List[Any] = []
        for messages in messages_list:
            mm_info = process_mm_info(messages, use_audio_in_video=False)
            # qwen-omni-utils commonly returns (audios, images, videos).
            if len(mm_info) == 3:
                a, i, v = mm_info
            else:
                a, i, v = mm_info[0], mm_info[1], mm_info[2]
            audios.extend(a or [])
            images.extend(i or [])
            videos.extend(v or [])

        try:
            inputs = self.processor(
                text=texts,
                audio=audios if audios else None,
                images=images if images else None,
                videos=videos if videos else None,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            )
        except TypeError:
            inputs = self.processor(
                text=texts,
                audios=audios if audios else None,
                images=images if images else None,
                videos=videos if videos else None,
                return_tensors="pt",
                padding=True,
            )
        return self._move_inputs_to_device(inputs)

    def _prepare_text_inputs(self, messages_list: List[List[Dict[str, Any]]]) -> Any:
        texts = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_list]
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        return self._move_inputs_to_device(inputs)

    def generate(self, audio_path: str, system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
        return self.generate_batch([audio_path], system_prompt, [user_prompt], max_new_tokens)[0]

    def generate_batch(self, audio_paths: List[str], system_prompt: str, user_prompts: List[str], max_new_tokens: int) -> List[str]:
        if len(audio_paths) != len(user_prompts):
            raise ValueError("audio_paths and user_prompts must have the same length")
        if not audio_paths:
            return []
        try:
            messages_list = [self._audio_messages(path, system_prompt, prompt) for path, prompt in zip(audio_paths, user_prompts)]
            inputs = self._prepare_audio_inputs(messages_list)
            generated = self._generate_from_inputs(inputs, max_new_tokens)
            return self._decode_new_tokens(generated, inputs, len(audio_paths))
        except Exception as exc:
            if len(audio_paths) == 1:
                raise
            logging.warning("Batched Qwen2.5-Omni generation failed; falling back to per-sample generation: %s", exc)
            return [self.generate(path, system_prompt, prompt, max_new_tokens) for path, prompt in zip(audio_paths, user_prompts)]

    def generate_text(self, system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
        return self.generate_text_batch(system_prompt, [user_prompt], max_new_tokens)[0]

    def generate_text_batch(self, system_prompt: str, user_prompts: List[str], max_new_tokens: int) -> List[str]:
        if not user_prompts:
            return []
        messages_list = [self._text_messages(system_prompt, prompt) for prompt in user_prompts]
        inputs = self._prepare_text_inputs(messages_list)
        generated = self._generate_from_inputs(inputs, max_new_tokens)
        return self._decode_new_tokens(generated, inputs, len(user_prompts))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="public_val", choices=["train", "public_val", "private_test"])
    p.add_argument("--audio_root", type=Path, default=None)
    p.add_argument("--question_root", type=Path, default=r"/root/autodl-tmp/trident")
    p.add_argument("--prob_csv", type=Path, default=r"/root/project/audio-deepfake-main/audio-deepfake-main/evidence_files/val_predictions_epoch_15.csv")
    p.add_argument("--model_path", default=r"/root/autodl-tmp/Qwen2.5-Omni-3B")
    p.add_argument("--output_dir", type=Path, default=r"outputs/qwen25_omni_refactored/public_val")
    p.add_argument("--live_log_file", type=Path, default=None)
    p.add_argument("--threshold_json", type=Path, default=Path("/root/project/audio-deepfake-main/audio-deepfake-main/evidence_files/audio_thresholds.json"))
    p.add_argument("--prompt_template", type=Path, default=Path("/root/project/audio-deepfake-main/audio-deepfake-main/prompts/audio_prompt_templates.yaml"))
    p.add_argument("--input_source", default="audio_evidence", choices=["audio", "audio_evidence", "evidence"])
    p.add_argument("--answer_strategy", default="postprocess", choices=["prompt", "postprocess"])
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
    p.add_argument("--check_deps", action="store_true")
    p.add_argument("--save_prompt_examples", action="store_true")
    p.add_argument("--prompt_examples_per_task", type=int, default=2)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    base.setup_logging()
    raw_argv = sys.argv[1:] if argv is None else argv
    if "--check_deps" in raw_argv:
        ok, report = check_qwen25_omni_dependencies()
        print("\n".join(report))
        return 0 if ok else 1

    args = parse_args(argv)
    requested = base.TASKS if args.mode == "all" else [args.mode]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.live_log_file is None:
        args.live_log_file = output_dir / "live_generation.jsonl"
    args.live_log_file.parent.mkdir(parents=True, exist_ok=True)
    args.live_log_file.write_text("", encoding="utf-8")
    (output_dir / "missing_samples.json").write_text("[]\n", encoding="utf-8")

    thresholds = base.load_thresholds(args.threshold_json)
    audio_root = base.resolve_audio_root(args.split, args.audio_root)
    args.audio_root = audio_root
    base.write_run_config(args, output_dir, thresholds, audio_root)
    builder = PromptBuilder.from_yaml(args.prompt_template)
    audio_index = base.build_audio_index(audio_root)
    prob_index = base.read_probability_csv(args.prob_csv, thresholds)
    task_files = base.discover_question_files(args.question_root, args.split)

    inferencer = None
    if not args.disable_llm:
        inferencer = Qwen25OmniInferencer(args.model_path, device=args.device, dtype=args.dtype, temperature=args.temperature, top_p=args.top_p)
    else:
        logging.info("LLM disabled")

    expected: Dict[str, int] = {}
    for task in requested:
        files = task_files.get(task, [])
        if not files:
            base.write_jsonl(output_dir / f"{task}.jsonl", [])
            base.write_jsonl(output_dir / f"debug_{task}.jsonl", [])
            expected[task] = 0
            continue
        outputs = q2ref.process_task(task, files, args, thresholds, audio_index, prob_index, inferencer, builder)
        expected[task] = len(outputs)
    for task in base.TASKS:
        if task not in requested:
            base.write_jsonl(output_dir / f"{task}.jsonl", [])
            base.write_jsonl(output_dir / f"debug_{task}.jsonl", [])
            expected[task] = 0
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
