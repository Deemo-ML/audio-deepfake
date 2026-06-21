# TRIDENT Audio Ablation Patch

This patch adds six Omni ablation settings to `scripts/generate_all_audio_answers_qwen25_omni.py`:

- S1: `--input_source audio --answer_strategy prompt`
- S2: `--input_source audio --answer_strategy postprocess`
- S3: `--input_source audio_evidence --answer_strategy prompt`
- S4: `--input_source audio_evidence --answer_strategy postprocess`
- S5: `--input_source evidence --answer_strategy prompt`
- S6: `--input_source evidence --answer_strategy postprocess --disable_llm`

Prompts are stored in `prompts/audio_prompt_templates.yaml`.

Dependency: `pip install pyyaml`.
