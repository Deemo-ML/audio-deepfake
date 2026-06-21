"""Prompt construction utilities for TRIDENT audio answer generation.

This module keeps prompt construction outside the inference script so that the full prompt assembly process can be inspected, logged, and edited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - checked by from_yaml.
    yaml = None  # type: ignore[assignment]


AUDIO_ARTIFACTS = [
    "Clipping",
    "Hiss",
    "Buzz",
    "Pops",
    "Unnatural Prosody",
]


@dataclass
class PromptBundle:
    """Final prompts and a human-readable construction trace."""

    system_prompt: str
    user_prompt: str
    template_key: str
    evidence_text: str
    options_text: str
    required_label: str
    steps: List[Dict[str, str]] = field(default_factory=list)

    def as_debug_dict(self) -> Dict[str, Any]:
        return {
            "template_key": self.template_key,
            "required_label": self.required_label,
            "evidence_text": self.evidence_text,
            "options_text": self.options_text,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "steps": self.steps,
        }


def artifact_list_text(artifacts: List[str]) -> str:
    if not artifacts:
        return ""
    if len(artifacts) == 1:
        return artifacts[0]
    return ", ".join(artifacts[:-1]) + " and " + artifacts[-1]


def typeb_required_label(pred_fake: bool, detected_artifacts: List[str]) -> str:
    return "Likely Manipulated" if pred_fake or bool(detected_artifacts) else "Likely Authentic"


def format_options(options: Optional[Mapping[str, str]]) -> str:
    if not options:
        return ""
    return "\n".join(f"{letter}. {text}" for letter, text in options.items())


def render_template(template: str, **kwargs: Any) -> str:
    """Render a simple str.format template with None-safe values."""
    safe_kwargs = {key: "" if value is None else value for key, value in kwargs.items()}
    return template.format(**safe_kwargs)


def _preview(text: str, max_chars: int = 360) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


class PromptBuilder:
    """Build task-specific TRIDENT prompts from a YAML template file."""

    def __init__(self, templates: Mapping[str, Any]) -> None:
        self.templates = dict(templates)
        self.system_prompt = str(self.templates.get("system_prompt", "")).strip()
        self.artifact_definitions = str(self.templates.get("artifact_definitions", "")).strip()

    @classmethod
    def from_yaml(cls, path: Path) -> "PromptBuilder":
        if not path.exists():
            raise FileNotFoundError(f"Prompt template file not found: {path}")
        if yaml is None:
            raise RuntimeError("PyYAML is required for prompt templates. Install with: pip install pyyaml")
        with path.open("r", encoding="utf-8-sig") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Prompt template file must contain a YAML mapping: {path}")
        return cls(data)

    def build_structured_evidence(self, evidence: Any, thresholds: Mapping[str, Any]) -> str:
        """Build the evidence block inserted into prompt templates.

        The evidence object is duck-typed so this builder can be reused with either dataclasses or plain objects as long as it exposes prob_fake, pred_fake, artifact_probs, and detected_artifacts.
        """
        artifact_thresholds = thresholds.get("artifact_thresholds", {})
        fake_threshold = float(thresholds.get("fake_threshold", 0.55))
        detected_artifacts = list(getattr(evidence, "detected_artifacts", []) or [])
        artifact_probs = getattr(evidence, "artifact_probs", {}) or {}
        prob_fake = float(getattr(evidence, "prob_fake", 0.0))
        pred_fake = bool(getattr(evidence, "pred_fake", False))

        lines = [
            "Structured detector evidence:",
            f"- prob_fake: {prob_fake:.4f}",
            f"- fake_threshold: {fake_threshold:.4f}",
            "- thresholded authenticity decision from prob_fake: " + ("Likely Manipulated" if pred_fake else "Likely Authentic"),
            "- required Type-B final label: " + typeb_required_label(pred_fake, detected_artifacts),
            "- artifact probabilities:",
        ]
        for artifact in AUDIO_ARTIFACTS:
            prob = float(artifact_probs.get(artifact, 0.0))
            threshold = float(artifact_thresholds.get(artifact, 0.50))
            status = "detected" if artifact in detected_artifacts else "not_detected"
            lines.append(f"  - {artifact}: prob={prob:.4f}, threshold={threshold:.4f}, status={status}")

        detected = artifact_list_text(detected_artifacts) if detected_artifacts else "None"
        forbidden = [artifact for artifact in AUDIO_ARTIFACTS if artifact not in detected_artifacts]
        lines.append(f"Reportable artifact candidates: {detected}")
        lines.append(f"Forbidden artifacts: {artifact_list_text(forbidden) if forbidden else 'None'}")
        return "\n".join(lines)

    def _select_template(self, task: str, input_source: str, answer_strategy: str) -> tuple[str, str]:
        if answer_strategy == "prompt":
            key = f"prompt_generation.{input_source}.{task}"
            template = self.templates.get("prompt_generation", {}).get(input_source, {}).get(task)
        elif answer_strategy == "postprocess":
            if input_source in {"audio", "audio_evidence"}:
                key = f"observation.{input_source}"
                template = self.templates.get("observation", {}).get(input_source)
            elif input_source == "evidence":
                key = f"prompt_generation.evidence.{task}"
                template = self.templates.get("prompt_generation", {}).get("evidence", {}).get(task)
            else:
                raise ValueError(f"Unknown input_source: {input_source}")
        else:
            raise ValueError(f"Unknown answer_strategy: {answer_strategy}")

        if template is None:
            raise KeyError(f"Missing prompt template: {key}")
        return key, str(template)

    def build(
        self,
        *,
        task: str,
        input_source: str,
        answer_strategy: str,
        question: str,
        evidence: Any,
        thresholds: Mapping[str, Any],
        options: Optional[Mapping[str, str]] = None,
    ) -> PromptBundle:
        """Build the system/user prompt pair and construction trace."""
        evidence_text = self.build_structured_evidence(evidence, thresholds)
        options_text = format_options(options)
        detected_artifacts = list(getattr(evidence, "detected_artifacts", []) or [])
        required_label = typeb_required_label(bool(getattr(evidence, "pred_fake", False)), detected_artifacts)
        template_key, template = self._select_template(task, input_source, answer_strategy)

        rendered_task_prompt = render_template(
            template,
            question=question,
            evidence_text=evidence_text,
            options_text=options_text,
            required_label=required_label,
        )
        prompt_parts = [part for part in [self.artifact_definitions, rendered_task_prompt] if part.strip()]
        user_prompt = "\n\n".join(prompt_parts)

        steps = [
            {"name": "select_template", "detail": template_key, "preview": _preview(template)},
            {"name": "artifact_definitions", "detail": "Inserted shared artifact reference guide from YAML.", "preview": _preview(self.artifact_definitions)},
            {"name": "structured_evidence", "detail": "Rendered detector probabilities, reportable candidates, and forbidden artifacts.", "preview": _preview(evidence_text)},
            {"name": "task_variables", "detail": f"task={task}, input_source={input_source}, answer_strategy={answer_strategy}, required_label={required_label}", "preview": _preview(question)},
            {"name": "final_user_prompt", "detail": "Concatenated artifact definitions and rendered task template.", "preview": _preview(user_prompt)},
        ]
        return PromptBundle(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            template_key=template_key,
            evidence_text=evidence_text,
            options_text=options_text,
            required_label=required_label,
            steps=steps,
        )
