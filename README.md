# DOLPHIN Artifact

DOLPHIN is a source-guided neuro-symbolic agentic framework for inferring whether compile-time features are present in stripped COTS binaries. The system first localizes feature-related source semantics from build scripts and source code, then uses source-binary similarity, call-graph alignment, and an inference agent to audit prioritized decompiled functions.

Running the full DOLPHIN pipeline requires IDA Pro for binary disassembly and decompilation.

## Artifact Structure

```text
.
|-- README.md
|-- 1-dolphin/
`-- 2-experiment-data/
```

### `1-dolphin/`

The implementation directory for DOLPHIN.

Important subdirectories and files:

- `config/feature/`: configuration files for feature semantic localization and feature presence inference with the evaluated LLM backends.
- `evaluation/feature/`: scripts for the feature-localization and feature-recovery evaluation pipeline.
- `sweagent/`: agent framework code adapted for DOLPHIN's source and binary analysis workflows.
- `swereview/`: review and analysis utilities for inspecting prompts, trajectories, problems, and results.
- `swerex/`: execution/runtime support code used by the agent framework.
- `tools/`: command-line tools exposed to the agents, including search, editing, file-map, submission, registry, and feature-analysis helpers.
- `templates/`: HTML templates for rendering problems and trajectories.
- `pyproject.toml` and `uv.lock`: Python project metadata and locked dependencies.

### `2-experiment-data/`

The experiment data used to support DOLPHIN's evaluation.

Important subdirectories and files:

- `benchmark/benchmark.csv`: benchmark metadata for the evaluated C libraries, features, versions, feature flags, associated CVEs, and source-code impact statistics.
- `rq1/rq1-effectiveness-and-robustness.csv`: effectiveness and robustness results, including precision, recall, F1, and accuracy for DOLPHIN and baseline tools.
- `rq2/rq2-overhead-analysis.csv`: overhead measurements for Stage 1 and Stage 2, including time, token usage, and cost.
- `rq3/rq3-stage1-ablation.csv`: ablation data for feature-semantic localization.
- `rq3/rq2-stage2-ablation.csv`: ablation data for Stage 2 candidate retrieval, call-graph alignment, and audit budget settings.
- `rq4/android/android-complete-metadata.csv`: Android native-library case-study metadata for OpenSSL `asm` feature profiling.
- `rq4/android/example/`: detailed Android case-study logs demonstrating how DOLPHIN works, including logs, predictions, and result JSON files.
- `rq4/firmware/firmware-complete-metadta.csv`: firmware Heartbleed case-study metadata and DOLPHIN triage evidence.
- `rq4/firmware/example/`: detailed firmware case-study logs demonstrating how DOLPHIN works, including logs, prioritized paths, and result JSON files.

## Notes

- The top-level artifact is organized into implementation (`1-dolphin`) and experiment data (`2-experiment-data`).
- The CSV files are intended to make the evaluation results auditable without requiring reviewers to rerun every experiment.
- The example files under `rq4/android/example/` and `rq4/firmware/example/` provide concrete DOLPHIN outputs and detailed logs for the downstream Android and firmware case studies.
