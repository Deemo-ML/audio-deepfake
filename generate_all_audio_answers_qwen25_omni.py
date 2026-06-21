#!/usr/bin/env python
"""Generate all TRIDENT audio answers with Qwen2.5-Omni and detector evidence."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - keeps the script usable in minimal envs.
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable: Optional[Iterable[Any]] = None, total: Optional[int] = None, **_: Any):
            self.iterable = iterable
            self.total = total
            self.n = 0

        def __iter__(self) -> Iterable[Any]:
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        def __enter__(self) -> "tqdm":
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def update(self, n: int = 1) -> None:
            self.n += n

        def set_postfix(self, *_: Any, **__: Any) -> None:
            return None


AUDIO_ARTIFACTS = [
    "Clipping",
    "Hiss",
    "Buzz",
    "Pops",
    "Unnatural Prosody",
]

ARTIFACT_DEFINITIONS = {
    "Clipping": "harsh, fuzzy, crackling, or saturated sound that can occur when the audio is too loud",
    "Hiss": "high-frequency static noise, similar to a persistent shhhh sound",
    "Buzz": "low-frequency tone or hum, often resembling electrical interference",
    "Pops": "abrupt, short, sharp clicks or bursts that interrupt the audio",
    "Unnatural Prosody": "speech that sounds robotic, monotonous, flat, or rhythmically unnatural",
}

DEFAULT_THRESHOLDS = {
    "fake_threshold": 0.55,
    "artifact_thresholds": {artifact: 0.50 for artifact in AUDIO_ARTIFACTS},
}

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
TASKS = ["tfq", "mcq", "typea_oeq", "typeb_oeq"]
DEFAULT_AUDIO_ROOTS = {
    "train": Path(r"E:\data\trident\OEQ\train\media\audio"),
    "public_val": Path(r"E:\data\trident\OEQ\public_val\media\audio"),
    "private_test": Path(r"E:\data\trident\OEQ\private_test\media\audio"),
}

SYSTEM_PROMPT = """You are an audio forensic analyst for the TRIDENT audio deepfake detection task.

You will receive:
1. An audio file.
2. Structured evidence from an external audio artifact detector.

You must answer according to the requested task type.

Rules:
- Use the audio signal and the structured detector evidence together.
- Only use the following artifact names: Clipping, Hiss, Buzz, Pops, Unnatural Prosody.
- Do not mention artifacts whose status is not_detected.
- Do not invent unsupported artifacts.
- For TFQ, output only True or False.
- For MCQ, output only option letter(s), such as A, B, A,C, or E.
- For Type-A OEQ, provide a structured artifact analysis for detected audio artifacts.
- For Type-B OEQ, start with exactly Likely Authentic or Likely Manipulated, followed by one concise evidence sentence.
"""

@dataclass
class Evidence:
    sample_id: str
    prob_fake: float
    artifact_probs: Dict[str, float]
    pred_fake: bool
    detected_artifacts: List[str]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def normalize_sample_id(value: Any) -> str:
    text = str(value or "").strip().strip('"').strip("'")
    if not text:
        return ""
    text = text.replace("\\", "/")
    name = Path(text).name
    stem = Path(name).stem
    return stem or name


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        logging.warning("Could not parse float value %r; using %.4f", value, default)
        return default


def find_column(fieldnames: Iterable[str], aliases: Iterable[str]) -> Optional[str]:
    normalized = {normalize_key(name): name for name in fieldnames}
    for alias in aliases:
        key = normalize_key(alias)
        if key in normalized:
            return normalized[key]
    return None


def load_thresholds(path: Optional[Path]) -> Dict[str, Any]:
    thresholds = json.loads(json.dumps(DEFAULT_THRESHOLDS))
    if path and path.exists():
        with path.open("r", encoding="utf-8-sig") as f:
            loaded = json.load(f)
        thresholds["fake_threshold"] = float(loaded.get("fake_threshold", thresholds["fake_threshold"]))
        loaded_artifacts = loaded.get("artifact_thresholds", {})
        normalized_artifacts = {normalize_key(key): value for key, value in loaded_artifacts.items()}
        for artifact in AUDIO_ARTIFACTS:
            value = loaded_artifacts.get(artifact, normalized_artifacts.get(normalize_key(artifact), thresholds["artifact_thresholds"][artifact]))
            thresholds["artifact_thresholds"][artifact] = float(value)
    elif path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(thresholds, f, ensure_ascii=False, indent=2)
            f.write("\n")
        logging.warning("Threshold file %s did not exist; wrote defaults", path)
    return thresholds
def read_probability_csv(path: Path, thresholds: Dict[str, Any]) -> Dict[str, Evidence]:
    if not path.exists():
        raise FileNotFoundError(f"Probability CSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        fieldnames = reader.fieldnames
        sample_col = find_column(fieldnames, ["sample_id", "sample id", "id", "sample"])
        media_col = find_column(fieldnames, ["media_path", "media path", "audio_path", "audio path", "path", "file"])
        prob_col = find_column(fieldnames, ["prob_fake", "fake_prob", "p_fake", "probability_fake", "fake_probability", "score"])
        artifact_cols = {}
        for artifact in AUDIO_ARTIFACTS:
            snake = artifact.replace(" ", "_")
            artifact_cols[artifact] = find_column(
                fieldnames,
                [
                    artifact,
                    artifact.lower(),
                    snake,
                    snake.lower(),
                    f"prob_{artifact}",
                    f"prob_{snake}",
                    f"prob {artifact}",
                    f"p_{artifact}",
                    f"p_{snake}",
                    f"{artifact}_prob",
                    f"{snake}_prob",
                ],
            )
        logging.info("Probability CSV column mapping: prob_fake=%s, artifacts=%s", prob_col, artifact_cols)

        rows: Dict[str, Evidence] = {}
        for row_index, row in enumerate(reader, start=2):
            sample_candidates = []
            if sample_col:
                sample_candidates.append(normalize_sample_id(row.get(sample_col)))
            if media_col:
                sample_candidates.append(normalize_sample_id(row.get(media_col)))
            sample_candidates = [sample for sample in dict.fromkeys(sample_candidates) if sample]
            if not sample_candidates:
                logging.warning("Skipping CSV row %d without sample_id/media_path", row_index)
                continue
            sample_id = sample_candidates[0]

            prob_fake = safe_float(row.get(prob_col), 0.0) if prob_col else 0.0
            artifact_probs = {
                artifact: safe_float(row.get(col), 0.0) if col else 0.0
                for artifact, col in artifact_cols.items()
            }
            pred_fake = prob_fake >= float(thresholds["fake_threshold"])
            detected = [
                artifact
                for artifact in AUDIO_ARTIFACTS
                if artifact_probs[artifact] >= float(thresholds["artifact_thresholds"][artifact])
            ]
            evidence = Evidence(sample_id, prob_fake, artifact_probs, pred_fake, detected)
            for candidate in sample_candidates:
                if candidate in rows:
                    logging.warning("Duplicate probability row for sample_id=%s; keeping the last row", candidate)
                rows[candidate] = evidence
    logging.info("Loaded probability lookup keys for %d samples from %s", len(rows), path)
    return rows


def build_audio_index(audio_root: Path) -> Dict[str, Path]:
    if not audio_root.exists():
        raise FileNotFoundError(f"Audio root not found: {audio_root}")
    index: Dict[str, Path] = {}
    duplicates = 0
    for path in audio_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            sample_id = normalize_sample_id(path.name)
            if sample_id in index:
                duplicates += 1
                logging.warning("Duplicate audio stem %s: %s and %s", sample_id, index[sample_id], path)
            index[sample_id] = path
    logging.info("Indexed %d audio files under %s (%d duplicate stems)", len(index), audio_root, duplicates)
    return index


def read_json_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected object in JSONL at {path}:{line_no}")
                records.append(obj)
        return records

    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        for key in ["questions", "data", "items", "records", "samples"]:
            value = data.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
        list_values = [value for value in data.values() if isinstance(value, list)]
        if len(list_values) == 1:
            return [record for record in list_values[0] if isinstance(record, dict)]
    raise ValueError(f"Unsupported question file structure: {path}")


def read_csv_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def read_question_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return read_json_records(path)
    if suffix == ".csv":
        return read_csv_records(path)
    raise ValueError(f"Unsupported question file suffix: {path}")


def infer_trident_root(question_root: Path, split: str) -> Path:
    root = question_root.resolve()
    if root.name.lower() == split.lower() and root.parent.name.upper() in {"TFQ", "MCQ", "OEQ"}:
        return root.parent.parent
    if root.name.upper() in {"TFQ", "MCQ", "OEQ"}:
        return root.parent
    if (root / "TFQ").exists() or (root / "MCQ").exists() or (root / "OEQ").exists():
        return root
    if root.parent.name.upper() in {"TFQ", "MCQ", "OEQ"}:
        return root.parent.parent
    return root


def add_existing(paths: List[Path], candidates: Iterable[Path]) -> None:
    seen = {path.resolve() for path in paths if path.exists()}
    for candidate in candidates:
        if candidate.exists() and candidate.resolve() not in seen:
            paths.append(candidate)
            seen.add(candidate.resolve())


def discover_question_files(question_root: Path, split: str) -> Dict[str, List[Path]]:
    if not question_root.exists():
        raise FileNotFoundError(f"Question root not found: {question_root}")

    task_files = {"tfq": [], "mcq": [], "typea_oeq": [], "typeb_oeq": []}
    trident_root = infer_trident_root(question_root, split)
    logging.info("Using TRIDENT question root: %s", trident_root)

    tfq_dir = trident_root / "TFQ" / split
    mcq_dir = trident_root / "MCQ" / split
    oeq_dir = trident_root / "OEQ" / split
    add_existing(task_files["tfq"], [tfq_dir / "aud_001.json", tfq_dir / "aud_002.json"])
    add_existing(task_files["mcq"], [mcq_dir / "aud_001.json", mcq_dir / "aud_002.json"])
    add_existing(task_files["typea_oeq"], [oeq_dir / "manifest_audio.csv"])
    add_existing(task_files["typeb_oeq"], [oeq_dir / "manifest_audio.csv"])

    files = [
        p for p in question_root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".csv"}
    ]
    split_key = split.lower()
    for path in files:
        lower_parts = {part.lower() for part in path.parts}
        if split_key not in lower_parts:
            continue
        name = path.name.lower()
        stem = path.stem.lower()
        is_audio_json = stem.startswith("aud_") and path.suffix.lower() in {".json", ".jsonl"}
        if is_audio_json and "tfq" in lower_parts:
            add_existing(task_files["tfq"], [path])
        elif is_audio_json and "mcq" in lower_parts:
            add_existing(task_files["mcq"], [path])
        elif name == "manifest_audio.csv" and "oeq" in lower_parts:
            add_existing(task_files["typea_oeq"], [path])
            add_existing(task_files["typeb_oeq"], [path])

    for task, paths in task_files.items():
        logging.info("Discovered %d %s question file(s): %s", len(paths), task, [str(path) for path in paths])
    return task_files


def get_first_present(record: Dict[str, Any], keys: Iterable[str]) -> Any:
    normalized = {normalize_key(key): key for key in record.keys()}
    for wanted in keys:
        key = normalized.get(normalize_key(wanted))
        if key is not None:
            return record.get(key)
    return None


def get_question_text(record: Dict[str, Any]) -> str:
    value = get_first_present(record, ["question", "query", "prompt", "text", "question_text"])
    return str(value or "").strip()


def get_task_question_text(record: Dict[str, Any], task: str) -> str:
    if task == "typea_oeq":
        value = get_first_present(
            record,
            [
                "typea_question",
                "type_a_question",
                "type a question",
                "question_typea",
                "question_type_a",
                "typea_prompt",
                "type_a_prompt",
                "artifact_question",
                "artifact_prompt",
                "oeq_a",
                "typea",
                "type_a",
            ],
        )
        if value:
            return str(value).strip()
    elif task == "typeb_oeq":
        value = get_first_present(
            record,
            [
                "typeb_question",
                "type_b_question",
                "type b question",
                "question_typeb",
                "question_type_b",
                "typeb_prompt",
                "type_b_prompt",
                "authenticity_question",
                "detection_question",
                "authenticity_prompt",
                "oeq_b",
                "typeb",
                "type_b",
            ],
        )
        if value:
            return str(value).strip()
    value = get_question_text(record)
    if value:
        return value
    if task == "typea_oeq":
        return "Describe the observable audio deepfake artifacts."
    if task == "typeb_oeq":
        return "Determine whether the audio is authentic or manipulated."
    return ""


def get_record_id(record: Dict[str, Any]) -> str:
    value = get_first_present(record, ["id", "question_id", "qid", "uuid"])
    return str(value or "").strip()


def get_sample_id(record: Dict[str, Any]) -> str:
    value = get_first_present(record, ["sample_id", "sample id", "audio_id", "media_id"])
    sample_id = normalize_sample_id(value)
    if sample_id:
        return sample_id
    value = get_first_present(record, ["media_path", "audio_path", "path", "file"])
    sample_id = normalize_sample_id(value)
    if sample_id:
        return sample_id
    return normalize_sample_id(get_record_id(record))


def response_field(record: Dict[str, Any]) -> str:
    # The public schema varies across starter kits. Preserve an existing answer-like
    # field; otherwise use the safe default field requested by the prompt.
    if "response" in record:
        return "response"
    if "answer" in record:
        return "answer"
    return "response"


def normalize_artifact_name(text: str) -> Optional[str]:
    key = normalize_key(text)
    aliases = {
        "clipping": "Clipping",
        "clip": "Clipping",
        "hiss": "Hiss",
        "hissing": "Hiss",
        "buzz": "Buzz",
        "buzzing": "Buzz",
        "hum": "Buzz",
        "pops": "Pops",
        "pop": "Pops",
        "clicks": "Pops",
        "click": "Pops",
        "unnaturalprosody": "Unnatural Prosody",
        "prosody": "Unnatural Prosody",
        "unnaturalrhythm": "Unnatural Prosody",
        "monotonic": "Unnatural Prosody",
    }
    if key in aliases:
        return aliases[key]
    for artifact in AUDIO_ARTIFACTS:
        if normalize_key(artifact) in key:
            return artifact
    return None


def extract_artifact_from_tfq(question: str) -> Optional[str]:
    for artifact in AUDIO_ARTIFACTS:
        if normalize_key(artifact) in normalize_key(question):
            return artifact
    return normalize_artifact_name(question)


def split_option_text(text: str, default_letter: str) -> Tuple[str, str]:
    match = re.match(r"^\s*([A-Z])[\.\):：、-]\s*(.+)$", str(text).strip(), flags=re.IGNORECASE)
    if match:
        return match.group(1).upper(), match.group(2).strip()
    return default_letter, str(text).strip()


def parse_mcq_options(record: Dict[str, Any]) -> Dict[str, str]:
    options = get_first_present(record, ["options", "choices", "candidates"])
    parsed: Dict[str, str] = {}
    if isinstance(options, dict):
        for key, value in options.items():
            letter = str(key).strip().upper()
            if len(letter) > 1:
                maybe_letter, text = split_option_text(f"{key}. {value}", letter[0])
                parsed[maybe_letter] = text
            else:
                parsed[letter] = str(value).strip()
    elif isinstance(options, list):
        for idx, value in enumerate(options):
            default_letter = chr(ord("A") + idx)
            letter, text = split_option_text(str(value), default_letter)
            parsed[letter] = text

    for idx, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        if letter in parsed:
            continue
        value = None
        for key in (f"option_{letter}", f"option{letter}", letter, f"option_{letter.lower()}", letter.lower()):
            if key in record:
                value = record[key]
                break
        if value is not None:
            parsed[letter] = str(value).strip()

    return {letter: text for letter, text in sorted(parsed.items()) if re.fullmatch(r"[A-Z]", letter)}


def is_none_option(text: str) -> bool:
    key = normalize_key(text)
    return (
        "none" in key
        or "nooption" in key
        or "nocorrect" in key
        or "notpresent" in key
        or "allincorrect" in key
    )


def artifact_list_text(artifacts: List[str]) -> str:
    if not artifacts:
        return ""
    if len(artifacts) == 1:
        return artifacts[0]
    return ", ".join(artifacts[:-1]) + " and " + artifacts[-1]


def typeb_required_label(pred_fake: bool, detected_artifacts: List[str]) -> str:
    return "Likely Manipulated" if pred_fake or bool(detected_artifacts) else "Likely Authentic"

def typea_template(detected_artifacts: List[str]) -> str:
    if detected_artifacts:
        return f"Observable artifacts include {artifact_list_text(detected_artifacts)}. These cues provide evidence of audio manipulation."
    return "No strong artifact evidence is confidently localized."


def typeb_template(pred_fake: bool, detected_artifacts: List[str]) -> str:
    required_label = typeb_required_label(pred_fake, detected_artifacts)
    if required_label == "Likely Authentic":
        return "Likely Authentic. No clear manipulation artifacts are detected in the audio."
    if detected_artifacts:
        return (
            f"Likely Manipulated. The audio shows evidence of {artifact_list_text(detected_artifacts)}, "
            "which supports the deepfake decision."
        )
    return (
        "Likely Manipulated. The manipulation probability is above the decision threshold, "
        "but no specific artifact exceeds the evidence threshold, so the explanation remains conservative."
    )


def mentioned_artifacts(text: str) -> List[str]:
    norm = normalize_key(text)
    found = []
    for artifact in AUDIO_ARTIFACTS:
        artifact_key = normalize_key(artifact)
        if artifact_key in norm:
            found.append(artifact)
        elif artifact == "Unnatural Prosody" and "prosody" in norm:
            found.append(artifact)
    return found


def sanitize_oeq_response(
    response: str,
    detected_artifacts: List[str],
    task_type: str,
    pred_fake: bool,
) -> Tuple[str, bool]:
    response = re.sub(r"\s+", " ", (response or "").strip())
    fallback = typeb_template(pred_fake, detected_artifacts) if task_type == "typeb_oeq" else typea_template(detected_artifacts)
    if not response:
        return fallback, True
    unsupported = [artifact for artifact in mentioned_artifacts(response) if artifact not in detected_artifacts]
    if unsupported:
        logging.warning("Model mentioned unsupported artifacts for %s: %s", task_type, unsupported)
        return fallback, True
    if task_type == "typeb_oeq":
        expected = typeb_required_label(pred_fake, detected_artifacts)
        if not response.startswith(expected):
            return fallback, True
        wrong_label = "Likely Authentic" if expected == "Likely Manipulated" else "Likely Manipulated"
        if response.startswith(wrong_label):
            return fallback, True
    return response, False


def build_structured_evidence(evidence: Evidence, thresholds: Dict[str, Any]) -> str:
    lines = [
        "Structured detector evidence:",
        f"- prob_fake: {evidence.prob_fake:.4f}",
        f"- fake_threshold: {float(thresholds['fake_threshold']):.4f}",
        "- thresholded authenticity decision from prob_fake: "
        + ("Likely Manipulated" if evidence.pred_fake else "Likely Authentic"),
        "- required Type-B final label: " + typeb_required_label(evidence.pred_fake, evidence.detected_artifacts),
        "- artifact probabilities:",
    ]
    for artifact in AUDIO_ARTIFACTS:
        prob = evidence.artifact_probs[artifact]
        threshold = float(thresholds["artifact_thresholds"][artifact])
        status = "detected" if artifact in evidence.detected_artifacts else "not_detected"
        lines.append(f"  - {artifact}: prob={prob:.4f}, threshold={threshold:.4f}, status={status}")
    detected = artifact_list_text(evidence.detected_artifacts) if evidence.detected_artifacts else "None"
    lines.append(f"Detected artifact set: {detected}")
    return "\n".join(lines)


def format_options(options: Dict[str, str]) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in options.items())


def build_user_prompt(task: str, question: str, evidence: Evidence, thresholds: Dict[str, Any], options: Optional[Dict[str, str]] = None) -> str:
    evidence_text = build_structured_evidence(evidence, thresholds)
    definitions = "\n".join(f"- {name}: {desc}." for name, desc in ARTIFACT_DEFINITIONS.items())
    if task == "tfq":
        task_text = f"""Task: True/False Question (TFQ)
Question: {question}

Decision rule:
- Identify the artifact asked about in the question.
- Answer True only if that artifact is in the Detected artifact set.
- Answer False if the artifact is absent, ambiguous, or not in the allowed artifact list.

Output format:
Return exactly one token, with no punctuation or explanation:
True
or
False"""
    elif task == "mcq":
        task_text = f"""Task: Multiple Choice Question (MCQ)
Question: {question}
Options:
{format_options(options or {})}

Decision rule:
- Select every option whose text names an artifact in the Detected artifact set.
- Do not select options whose artifacts are not_detected.
- If no listed artifact option is detected, select the option meaning "None of the options are correct".
- If multiple options are selected, keep the option-letter order shown above.

Output format:
Return only uppercase option letter(s), separated by commas with no spaces.
Valid examples: A, B, A,C, E"""
    elif task == "typea_oeq":
        task_text = f"""Task: Type-A Open-Ended Question (known DeepFake audio)
Question: {question}

Context:
This is a DeepFake sample generated by an advanced DeepFake technique.
Your task is to perform a detailed audio-forensics analysis of artifacts in this sample that seem inauthentic.

Artifact reference guide:
- Clipping: harsh, fuzzy, crackling, or saturated sound when the audio is too loud.
- Hiss: high-frequency static noise, similar to a persistent shhhh sound.
- Buzz: low-frequency tone or hum, typically caused by electrical interference.
- Pops: abrupt, short, sharp clicks or bursts that interrupt the audio.
- Unnatural Prosody: speech that sounds robotic, monotonous, flat, or rhythmically unnatural.

Guidelines:
- Be thorough: cover every artifact in the Detected artifact set.
- Be accurate: base the analysis on the audio and the structured detector evidence.
- Avoid false positives: do not identify authentic features as inauthentic.
- Mention only artifacts whose status is detected.
- Do not mention artifacts whose status is not_detected.
- If no artifact is detected, state that no strong artifact evidence is confidently localized.

Output format:
Use clear headings, one heading per detected artifact.
Under each heading, write a brief Reason explaining why that artifact is suspicious in audio-forensics terms.
No JSON. No markdown tables."""
    elif task == "typeb_oeq":
        task_text = f"""Task: Type-B Open-Ended Question (authenticity and artifact analysis)
Question: {question}

Role:
You are a DeepFake forensics analyst. Analyze the provided audio sample and identify artifacts that indicate it may be inauthentic or manipulated.

Artifact reference guide:
- Clipping: harsh, fuzzy, crackling, or saturated sound when the audio is too loud.
- Hiss: high-frequency static noise, similar to a persistent shhhh sound.
- Buzz: low-frequency tone or hum, typically caused by electrical interference.
- Pops: abrupt, short, sharp clicks or bursts that interrupt the audio.
- Unnatural Prosody: speech that sounds robotic, monotonous, flat, or rhythmically unnatural.

Decision rule:
- Follow the "required Type-B final label" in the structured detector evidence exactly.
- If the Detected artifact set is not None, the Overall Assessment must be Likely Manipulated, even if prob_fake alone is below fake_threshold.
- If prob_fake is at or above fake_threshold, the Overall Assessment must be Likely Manipulated.
- Use Likely Authentic only when prob_fake is below fake_threshold and the Detected artifact set is None.
- If you output Likely Authentic, do not list Buzz, Hiss, Clipping, Pops, or Unnatural Prosody as findings.
- Base artifact claims only on artifacts in the Detected artifact set.
- Do not mention artifacts whose status is not_detected.
- If uncertain or no artifact is detected, state the uncertainty conservatively and do not invent evidence.

Consistency rule:
- Never write "Likely Authentic" together with detected artifact findings.
- If any artifact finding is present, the response must start with "Likely Manipulated."

Output format:
Start the response exactly with one assessment sentence:
Likely Authentic. <brief evidence sentence>
or
Likely Manipulated. <brief evidence sentence>

Then write Artifact Findings.
- If the label is Likely Manipulated and artifacts are detected, list each detected artifact with Title and Reason.
- If the label is Likely Authentic, write: Artifact Findings: No detected audio deepfake artifacts.
No JSON. No markdown tables."""
    else:
        raise ValueError(f"Unknown task: {task}")
    return "\n\n".join(["Allowed audio artifacts:", definitions, evidence_text, task_text])


def final_tfq_answer(question: str, detected_artifacts: List[str]) -> Tuple[str, bool]:
    artifact = extract_artifact_from_tfq(question)
    if artifact is None:
        logging.warning("Could not identify TFQ artifact from question: %s", question)
        return "False", True
    return ("True" if artifact in detected_artifacts else "False"), False


def final_mcq_answer(options: Dict[str, str], detected_artifacts: List[str]) -> Tuple[str, bool]:
    selected = []
    none_letter = None
    for letter, text in options.items():
        if is_none_option(text):
            none_letter = letter
            continue
        artifact = normalize_artifact_name(text)
        if artifact and artifact in detected_artifacts:
            selected.append(letter)
    if selected:
        return ",".join(selected), False
    if none_letter:
        return none_letter, False
    logging.warning("No MCQ artifact option matched and no explicit none option exists")
    return "", True

def check_qwen_omni_dependencies() -> Tuple[bool, List[str]]:
    import importlib

    results: List[str] = []
    ok = True
    for module_name in ["torch", "transformers", "accelerate", "qwen_omni_utils"]:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            results.append(f"{module_name}: OK version={version}")
        except Exception as exc:
            ok = False
            results.append(f"{module_name}: FAIL {type(exc).__name__}: {exc}")

    try:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor  # noqa: F401
        results.append("transformers.Qwen2_5Omni*: OK")
    except Exception as exc:
        ok = False
        results.append(f"transformers.Qwen2_5Omni*: FAIL {type(exc).__name__}: {exc}")

    try:
        from qwen_omni_utils import process_mm_info  # noqa: F401
        results.append("qwen_omni_utils.process_mm_info: OK")
    except Exception as exc:
        ok = False
        results.append(f"qwen_omni_utils.process_mm_info: FAIL {type(exc).__name__}: {exc}")

    return ok, results


class QwenOmniAudioInferencer:
    def __init__(self, model_path: str, device: str = "cuda", dtype: str = "bf16", temperature: float = 0.0, top_p: float = 1.0):
        ok, dependency_report = check_qwen_omni_dependencies()
        if not ok:
            raise RuntimeError(
                "Failed to import Qwen2.5-Omni dependencies:\n"
                + "\n".join(f"- {line}" for line in dependency_report)
                + "\nInstall compatible versions of torch, transformers, accelerate, and qwen-omni-utils, then retry."
            )
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        from qwen_omni_utils import process_mm_info

        if device == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but unavailable; falling back to CPU")
            device = "cpu"

        dtype_map = {
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

        kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype}
        if device == "cuda":
            kwargs["device_map"] = "auto"

        self.torch = torch
        self.process_mm_info = process_mm_info
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, **kwargs)
        if device != "cuda":
            self.model.to(device)
        self.model.eval()

    def _processor_call(self, text: str, messages: List[Dict[str, Any]]) -> Any:
        mm_result = self.process_mm_info(messages, use_audio_in_video=False)
        audio_inputs = image_inputs = video_inputs = None
        if isinstance(mm_result, tuple):
            if len(mm_result) == 3:
                audio_inputs, image_inputs, video_inputs = mm_result
            elif len(mm_result) == 2:
                image_inputs, video_inputs = mm_result
            elif len(mm_result) == 1:
                audio_inputs = mm_result[0]
        else:
            audio_inputs = mm_result

        kwargs: Dict[str, Any] = {"text": [text], "return_tensors": "pt", "padding": True, "use_audio_in_video": False}
        if image_inputs is not None:
            kwargs["images"] = image_inputs
        if video_inputs is not None:
            kwargs["videos"] = video_inputs
        if audio_inputs is not None:
            kwargs["audio"] = audio_inputs
            try:
                return self.processor(**kwargs)
            except TypeError:
                kwargs["audios"] = kwargs.pop("audio")
                return self.processor(**kwargs)
        return self.processor(**kwargs)

    def generate(self, audio_path: str, system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_path)},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor_call(text, messages)
        inputs = inputs.to(self.device) if hasattr(inputs, "to") else inputs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "return_audio": False,
            "use_audio_in_video": False,
        }
        if self.temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": self.temperature, "top_p": self.top_p})
        with self.torch.inference_mode():
            try:
                generated = self.model.generate(**inputs, **gen_kwargs)
            except TypeError:
                gen_kwargs.pop("return_audio", None)
                gen_kwargs.pop("use_audio_in_video", None)
                generated = self.model.generate(**inputs, **gen_kwargs)
        if isinstance(generated, tuple):
            generated = generated[0]
        input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        output_ids = generated[:, input_len:] if input_len else generated
        decoded = self.processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return decoded[0].strip() if decoded else ""

    def generate_batch(
        self,
        audio_paths: List[str],
        system_prompt: str,
        user_prompts: List[str],
        max_new_tokens: int,
    ) -> List[str]:
        if len(audio_paths) != len(user_prompts):
            raise ValueError("audio_paths and user_prompts must have the same length")
        if not audio_paths:
            return []
        if len(audio_paths) == 1:
            return [self.generate(audio_paths[0], system_prompt, user_prompts[0], max_new_tokens)]

        message_batches = []
        for audio_path, user_prompt in zip(audio_paths, user_prompts):
            message_batches.append(
                [
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": str(audio_path)},
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ]
            )

        try:
            texts = [
                self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in message_batches
            ]
            mm_result = self.process_mm_info(message_batches, use_audio_in_video=False)
            audio_inputs = image_inputs = video_inputs = None
            if isinstance(mm_result, tuple):
                if len(mm_result) == 3:
                    audio_inputs, image_inputs, video_inputs = mm_result
                elif len(mm_result) == 2:
                    image_inputs, video_inputs = mm_result
                elif len(mm_result) == 1:
                    audio_inputs = mm_result[0]
            else:
                audio_inputs = mm_result

            kwargs: Dict[str, Any] = {
                "text": texts,
                "return_tensors": "pt",
                "padding": True,
                "use_audio_in_video": False,
            }
            if image_inputs is not None:
                kwargs["images"] = image_inputs
            if video_inputs is not None:
                kwargs["videos"] = video_inputs
            if audio_inputs is not None:
                kwargs["audio"] = audio_inputs

            try:
                inputs = self.processor(**kwargs)
            except TypeError:
                if "audio" in kwargs:
                    kwargs["audios"] = kwargs.pop("audio")
                    try:
                        inputs = self.processor(**kwargs)
                    except TypeError:
                        kwargs.pop("use_audio_in_video", None)
                        inputs = self.processor(**kwargs)
                else:
                    kwargs.pop("use_audio_in_video", None)
                    inputs = self.processor(**kwargs)

            inputs = inputs.to(self.device) if hasattr(inputs, "to") else inputs
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "return_audio": False,
                "use_audio_in_video": False,
            }
            if self.temperature > 0:
                gen_kwargs.update({"do_sample": True, "temperature": self.temperature, "top_p": self.top_p})
            with self.torch.inference_mode():
                try:
                    generated = self.model.generate(**inputs, **gen_kwargs)
                except TypeError:
                    gen_kwargs.pop("return_audio", None)
                    gen_kwargs.pop("use_audio_in_video", None)
                    generated = self.model.generate(**inputs, **gen_kwargs)
            if isinstance(generated, tuple):
                generated = generated[0]

            if "attention_mask" in inputs:
                input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
            else:
                input_lengths = [inputs["input_ids"].shape[-1]] * len(audio_paths)
            output_ids = [generated[i, int(input_lengths[i]) :] for i in range(len(audio_paths))]
            decoded = self.processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if len(decoded) != len(audio_paths):
                raise RuntimeError(f"Batch decode returned {len(decoded)} items for {len(audio_paths)} inputs")
            return [item.strip() for item in decoded]
        except Exception as exc:
            logging.warning("Batched Qwen generation failed; falling back to per-sample generation: %s", exc)
            return [
                self.generate(audio_path, system_prompt, user_prompt, max_new_tokens)
                for audio_path, user_prompt in zip(audio_paths, user_prompts)
            ]


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_live_log(handle: Any, record: Dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def validate_or_collect_missing(
    records: List[Dict[str, Any]],
    audio_index: Dict[str, Path],
    probability_index: Dict[str, Evidence],
    task: str,
) -> List[Dict[str, Any]]:
    missing = []
    for record in records:
        sample_id = get_sample_id(record)
        item = {
            "task": task,
            "id": get_record_id(record) or sample_id,
            "sample_id": sample_id,
            "question": get_task_question_text(record, task),
            "missing_audio": sample_id not in audio_index,
            "missing_probabilities": sample_id not in probability_index,
        }
        if item["missing_audio"] or item["missing_probabilities"]:
            missing.append(item)
    return missing


def max_tokens_for_task(task: str, args: argparse.Namespace) -> int:
    task_overrides = {
        "tfq": args.tfq_max_new_tokens,
        "mcq": args.mcq_max_new_tokens,
        "typea_oeq": args.typea_max_new_tokens,
        "typeb_oeq": args.typeb_max_new_tokens,
    }
    value = task_overrides.get(task)
    if value is None:
        if task == "tfq":
            return min(args.max_new_tokens, 8)
        if task == "mcq":
            return min(args.max_new_tokens, 16)
        return args.max_new_tokens
    if value < 1:
        raise ValueError(f"max_new_tokens for {task} must be positive, got {value}")
    return value


def iter_batches(records: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    batch_size = max(1, int(batch_size))
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def process_task(
    task: str,
    question_files: List[Path],
    args: argparse.Namespace,
    thresholds: Dict[str, Any],
    audio_index: Dict[str, Path],
    probability_index: Dict[str, Evidence],
    inferencer: Optional[QwenOmniAudioInferencer],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in question_files:
        loaded = read_question_records(path)
        logging.info("Loaded %d records from %s", len(loaded), path)
        records.extend(loaded)
    limit_candidates = []
    if args.first_n_questions is not None:
        if args.first_n_questions < 0:
            raise ValueError("--first_n_questions must be non-negative")
        limit_candidates.append(args.first_n_questions)
    if args.dry_run:
        limit_candidates.append(args.num_samples)
    if limit_candidates:
        limit = min(limit_candidates)
        logging.info("Limiting %s to first %d question(s)", task, limit)
        records = records[:limit]

    missing = validate_or_collect_missing(records, audio_index, probability_index, task)
    if missing:
        missing_path = Path(args.output_dir) / "missing_samples.json"
        existing = []
        if missing_path.exists():
            with missing_path.open("r", encoding="utf-8-sig") as f:
                existing = json.load(f)
        write_data = existing + missing
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        with missing_path.open("w", encoding="utf-8") as f:
            json.dump(write_data, f, ensure_ascii=False, indent=2)
        if not args.allow_missing:
            raise RuntimeError(f"{len(missing)} {task} sample(s) are missing audio/probabilities; see {missing_path}")
        logging.warning("Skipping %d missing %s sample(s)", len(missing), task)

    outputs = []
    debug_records = []
    batch_size = max(1, int(args.batch_size))
    task_max_new_tokens = max_tokens_for_task(task, args)
    num_batches = (len(records) + batch_size - 1) // batch_size if records else 0
    skipped_count = 0
    logging.info(
        "Processing %s with batch_size=%d, max_new_tokens=%d (%d batch(es))",
        task,
        batch_size,
        task_max_new_tokens,
        num_batches,
    )

    live_log_path = Path(args.live_log_file) if args.live_log_file else None
    live_log_handle = live_log_path.open("a", encoding="utf-8", newline="\n") if live_log_path else None
    try:
        with tqdm(total=len(records), desc=f"{task}", unit="sample", dynamic_ncols=True) as progress:
            for batch_idx, batch_records in enumerate(iter_batches(records, batch_size), start=1):
                batch_items: List[Dict[str, Any]] = []
                for record in batch_records:
                    sample_id = get_sample_id(record)
                    if sample_id not in audio_index or sample_id not in probability_index:
                        skipped_count += 1
                        continue
                    audio_path = audio_index[sample_id]
                    evidence = probability_index[sample_id]
                    question = get_task_question_text(record, task)
                    options = parse_mcq_options(record) if task == "mcq" else None
                    prompt = build_user_prompt(task, question, evidence, thresholds, options)
                    batch_items.append(
                        {
                            "record": record,
                            "sample_id": sample_id,
                            "audio_path": audio_path,
                            "evidence": evidence,
                            "question": question,
                            "options": options,
                            "prompt": prompt,
                        }
                    )

                raw_model_responses = [""] * len(batch_items)
                if inferencer is not None and batch_items:
                    try:
                        logging.info(
                            "Generating %s batch %d/%d with %d sample(s): %s",
                            task,
                            batch_idx,
                            num_batches,
                            len(batch_items),
                            [str(item["sample_id"]) for item in batch_items],
                        )
                        raw_model_responses = inferencer.generate_batch(
                            [str(item["audio_path"]) for item in batch_items],
                            SYSTEM_PROMPT,
                            [str(item["prompt"]) for item in batch_items],
                            task_max_new_tokens,
                        )
                        logging.info("Finished %s batch %d/%d", task, batch_idx, num_batches)
                    except Exception as exc:
                        if not args.rule_fallback:
                            raise
                        logging.warning("Batch model generation failed for task=%s: %s", task, exc)
                        raw_model_responses = [""] * len(batch_items)

                for item, raw_model_response in zip(batch_items, raw_model_responses):
                    record = item["record"]
                    sample_id = str(item["sample_id"])
                    audio_path = item["audio_path"]
                    evidence = item["evidence"]
                    question = str(item["question"])
                    options = item["options"]
                    prompt = str(item["prompt"])

                    used_fallback = False
                    if task == "tfq":
                        final_response, used_fallback = final_tfq_answer(question, evidence.detected_artifacts)
                    elif task == "mcq":
                        final_response, used_fallback = final_mcq_answer(options or {}, evidence.detected_artifacts)
                    elif task == "typea_oeq":
                        candidate = raw_model_response if raw_model_response and args.rule_fallback else raw_model_response
                        if not candidate or args.disable_llm:
                            candidate = typea_template(evidence.detected_artifacts)
                            used_fallback = True
                        final_response, sanitized = sanitize_oeq_response(candidate, evidence.detected_artifacts, task, evidence.pred_fake)
                        used_fallback = used_fallback or sanitized
                    elif task == "typeb_oeq":
                        candidate = raw_model_response if raw_model_response and args.rule_fallback else raw_model_response
                        if not candidate or args.disable_llm:
                            candidate = typeb_template(evidence.pred_fake, evidence.detected_artifacts)
                            used_fallback = True
                        final_response, sanitized = sanitize_oeq_response(candidate, evidence.detected_artifacts, task, evidence.pred_fake)
                        used_fallback = used_fallback or sanitized
                    else:
                        raise ValueError(f"Unknown task: {task}")

                    output_record = dict(record)
                    output_record.setdefault("id", get_record_id(record) or sample_id)
                    output_record.setdefault("sample_id", sample_id)
                    output_record[response_field(record)] = final_response
                    outputs.append(output_record)

                    debug_record = {
                        "task": task,
                        "id": get_record_id(record) or sample_id,
                        "sample_id": sample_id,
                        "audio_path": str(audio_path),
                        "question": question,
                        "prob_fake": evidence.prob_fake,
                        "fake_threshold": float(thresholds["fake_threshold"]),
                        "pred_fake": evidence.pred_fake,
                        "artifact_probs": evidence.artifact_probs,
                        "artifact_thresholds": thresholds["artifact_thresholds"],
                        "detected_artifacts": evidence.detected_artifacts,
                        "prompt": prompt,
                        "raw_model_response": raw_model_response,
                        "final_response": final_response,
                        "used_fallback": used_fallback,
                    }
                    debug_records.append(debug_record)
                    if live_log_handle is not None:
                        append_live_log(
                            live_log_handle,
                            {
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                                "task": task,
                                "batch": batch_idx,
                                "id": debug_record["id"],
                                "sample_id": sample_id,
                                "audio_path": str(audio_path),
                                "question": question,
                                "prob_fake": evidence.prob_fake,
                                "pred_fake": evidence.pred_fake,
                                "detected_artifacts": evidence.detected_artifacts,
                                "raw_model_response": raw_model_response,
                                "final_response": final_response,
                                "used_fallback": used_fallback,
                            },
                        )

                progress.update(len(batch_records))
                progress.set_postfix(
                    batch=f"{batch_idx}/{num_batches}",
                    batch_size=batch_size,
                    written=len(outputs),
                    skipped=skipped_count,
                    refresh=True,
                )

    finally:
        if live_log_handle is not None:
            live_log_handle.close()

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / f"{task}.jsonl", outputs)
    write_jsonl(output_dir / f"debug_{task}.jsonl", debug_records)
    logging.info("Wrote %d %s answers to %s", len(outputs), task, output_dir / f"{task}.jsonl")
    return outputs

def resolve_audio_root(split: str, audio_root: Optional[Path]) -> Path:
    if audio_root is not None:
        return audio_root
    try:
        return DEFAULT_AUDIO_ROOTS[split]
    except KeyError as exc:
        raise ValueError(f"Unsupported split for default audio root: {split}") from exc


def write_run_config(args: argparse.Namespace, output_dir: Path, thresholds: Dict[str, Any], audio_root: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "resolved_audio_root": str(audio_root),
        "tasks": TASKS if args.mode == "all" else [args.mode],
        "thresholds": thresholds,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            records.append(obj)
    return records


def validate_outputs(output_dir: Path, expected_counts: Dict[str, int]) -> None:
    for task in TASKS:
        path = output_dir / f"{task}.jsonl"
        debug_path = output_dir / f"debug_{task}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing output file: {path}")
        records = read_jsonl_file(path)
        if len(records) != expected_counts.get(task, 0):
            raise ValueError(
                f"Output count mismatch for {task}: got {len(records)}, expected {expected_counts.get(task, 0)}"
            )

        debug_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if debug_path.exists():
            for debug in read_jsonl_file(debug_path):
                debug_by_key[(str(debug.get("id", "")), str(debug.get("sample_id", "")))] = debug

        for idx, record in enumerate(records, start=1):
            response = str(record.get(response_field(record), ""))
            if task == "tfq" and response not in {"True", "False"}:
                raise ValueError(f"Invalid TFQ response in {path}:{idx}: {response!r}")
            if task == "mcq" and not re.fullmatch(r"[A-Z](,[A-Z])*", response):
                raise ValueError(f"Invalid MCQ response in {path}:{idx}: {response!r}")
            if task == "typeb_oeq" and not (
                response.startswith("Likely Authentic") or response.startswith("Likely Manipulated")
            ):
                raise ValueError(f"Invalid Type-B response in {path}:{idx}: {response!r}")
            if task == "typea_oeq":
                key = (str(record.get("id", "")), str(record.get("sample_id", "")))
                detected = debug_by_key.get(key, {}).get("detected_artifacts", [])
                unsupported = [artifact for artifact in mentioned_artifacts(response) if artifact not in detected]
                if unsupported:
                    raise ValueError(f"Unsupported artifact(s) in {path}:{idx}: {unsupported}")
    logging.info("Validated output JSONL files in %s", output_dir)

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="public_val", choices=["train", "public_val", "private_test"])
    parser.add_argument("--audio_root", type=Path, default=None, help="Optional audio root; inferred from --split when omitted.")
    parser.add_argument("--question_root", type=Path, default=r"E:\data\trident")
    parser.add_argument("--prob_csv", type=Path, default=r"val_predictions_epoch_15.csv")
    parser.add_argument("--model_path", default=r"E:\model\Qwen2.5-Omni-3B")
    parser.add_argument("--output_dir", type=Path, default=r"outputs/qwen25_omni_audio/public_val")
    parser.add_argument("--live_log_file", type=Path, default=None, help="Realtime JSONL log file for generated results. Defaults to output_dir/live_generation.jsonl.")
    parser.add_argument("--threshold_json", type=Path, default=Path("configs/audio_thresholds.json"))
    parser.add_argument("--mode", default="typeb_oeq", choices=["all", "tfq", "mcq", "typea_oeq", "typeb_oeq"])
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Default max_new_tokens used when a task-specific value is not set.")
    parser.add_argument("--tfq_max_new_tokens", type=int, default=8, help="Override max_new_tokens for TFQ.")
    parser.add_argument("--mcq_max_new_tokens", type=int, default=16, help="Override max_new_tokens for MCQ.")
    parser.add_argument("--typea_max_new_tokens", type=int, default=32, help="Override max_new_tokens for Type-A OEQ.")
    parser.add_argument("--typeb_max_new_tokens", type=int, default=128, help="Override max_new_tokens for Type-B OEQ.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--rule_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable_llm", action="store_true", help="Use deterministic template/rule answers only.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--first_n_questions", type=int, default=10, help="Only process the first N questions per task.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of samples to send through generation per batch.")
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32", "auto"])
    parser.add_argument("--check_deps", action="store_true", help="Only check Qwen2.5-Omni Python dependencies and exit.")
    return parser.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    raw_argv = sys.argv[1:] if argv is None else argv
    if "--check_deps" in raw_argv:
        ok, dependency_report = check_qwen_omni_dependencies()
        for line in dependency_report:
            print(line)
        return 0 if ok else 1
    args = parse_args(argv)
    logging.info("Selected split: %s", args.split)
    requested_tasks = TASKS if args.mode == "all" else [args.mode]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.live_log_file is None:
        args.live_log_file = output_dir / "live_generation.jsonl"
    else:
        args.live_log_file = Path(args.live_log_file)
    args.live_log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.live_log_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write("")
    logging.info("Realtime generation log: %s", args.live_log_file)
    missing_path = output_dir / "missing_samples.json"
    with missing_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
        f.write("\n")

    thresholds = load_thresholds(args.threshold_json)
    audio_root = resolve_audio_root(args.split, args.audio_root)
    args.audio_root = audio_root
    write_run_config(args, output_dir, thresholds, audio_root)

    audio_index = build_audio_index(audio_root)
    probability_index = read_probability_csv(args.prob_csv, thresholds)
    task_files = discover_question_files(args.question_root, args.split)

    inferencer = None
    if not args.disable_llm:
        if not args.model_path:
            raise ValueError("--model_path is required unless --disable_llm is set")
        inferencer = QwenOmniAudioInferencer(
            args.model_path,
            device=args.device,
            dtype=args.dtype,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    else:
        logging.info("LLM disabled; generating deterministic evidence-constrained answers")

    expected_counts: Dict[str, int] = {}
    for task in requested_tasks:
        files = task_files.get(task, [])
        if not files:
            logging.warning("No question files discovered for task=%s; writing empty output", task)
            write_jsonl(output_dir / f"{task}.jsonl", [])
            write_jsonl(output_dir / f"debug_{task}.jsonl", [])
            expected_counts[task] = 0
            continue
        outputs = process_task(task, files, args, thresholds, audio_index, probability_index, inferencer)
        expected_counts[task] = len(outputs)

    for task in TASKS:
        if task not in requested_tasks:
            write_jsonl(output_dir / f"{task}.jsonl", [])
            write_jsonl(output_dir / f"debug_{task}.jsonl", [])
            expected_counts[task] = 0

    validate_outputs(output_dir, expected_counts)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        logging.error("%s", exc)
        raise SystemExit(1)
