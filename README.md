# RCR EvidenceBinding ablations

An experiment layer over the authoritative [Proposed repository](https://github.com/longvu2005/referential-composition-retrieval), pinned as the Git submodule `third_party/rcr`. The original method is still named `proposed`. All B1–B3 algorithms, overlays, tests and experiment outputs live in this repository.

| ID | `--variant` | Grounding EvidenceBinding | Context EvidenceBinding | Composition, matching, TargetPersonBuilder, FineReasoner, FAFA |
| --- | --- | --- | --- | --- |
| B0 | `proposed` | Original | Original | Original |
| B1 | `no_grounding_binding` | Masked-mean text/person MLP | Original | Original |
| B2 | `no_context_binding` | Original | Direct aligned person semantics | Original |
| B3 | `no_both_bindings` | B1 replacement | B2 replacement | Original |

B0 is the original `RCRModel`, with its original construction order, parameter names and initialization. It loads existing Proposed inference checkpoints. B1–B3 are independently trained models; a B0 checkpoint is not an ablation initialization or resume checkpoint.

## Setup

Use Python 3.11 or 3.12 and the original dependency environments:

```bash
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
python -m pip install -e 'third_party/rcr[proposed,dev]' -e '.[dev]'
```

The local integration commit must be published in Proposed before a fresh remote clone can fetch that submodule pin. During review, keep this existing checkout. The temporary branch is `feature/evidence-binding-ablations` inside `third_party/rcr`; no branch per variant is needed. See [integration steps](docs/integration.md) before publishing either repository.

Prepare the original dataset, detector, CLIP, native FAFA assets and feature caches using [Proposed's setup instructions](third_party/rcr/README.md). Existing compatible frozen caches are shared by all four models. Backbone downloads and GPU access are unnecessary for the CPU tests below.

## Training and resuming

```bash
python -m rcr_ablations.run train --variant proposed --seed 0
python -m rcr_ablations.run train --variant no_grounding_binding --seed 0
python -m rcr_ablations.run train --variant no_context_binding --seed 0
python -m rcr_ablations.run train --variant no_both_bindings --seed 0
```

These commands inherit `third_party/rcr/configs/proposed.yaml` and the corresponding one-key overlay in `configs/variants/`. They retain the original epochs, optimizer, schedule, batch sizes, sampling, objectives, encoders, calibration grid, seeds and `top_m=500`. Training uses train data; validation selects the checkpoint and calibrates the scores. It never evaluates test.

Runs are separated by variant and seed:

```text
runs/ablations/<variant>/seed<seed>/
  config.yaml       # Complete initial effective configuration
  manifest.json     # Effective checkpoint config, code/input/runtime fingerprints
  last.pt           # Model, text projection, optimizer, scaler, RNG and progress
  best.pt           # Validation-selected inference checkpoint
  selected.yaml     # Checkpoint-bound validation-selected scoring contract
  evaluation/       # Original validation and calibration artifacts
```

Use `--seed 1`, etc. for additional runs. Existing training directories are rejected. Resume the same run using the same numerical settings and source:

```bash
python -m rcr_ablations.run train --variant no_grounding_binding --seed 0 --resume
python -m rcr_ablations.run train \
  --config runs/ablations/no_grounding_binding/seed0/config.yaml \
  --resume runs/ablations/no_grounding_binding/seed0/last.pt
```

Inference-only and legacy checkpoints cannot resume training exactly. Variant/class mismatches and changes to numerical configuration, source, caches, input fingerprints or runtime are rejected by the original resume protocol. Models load strictly; these experiments do not introduce `strict=False`.

## Validation, retrieval and calibration

Checkpoint metadata reconstructs the selected model; `--variant` is optional when using `--checkpoint` or a saved effective configuration. An explicit conflicting variant is rejected.

```bash
# Re-run validation retrieval and official evaluation.
python -m rcr_ablations.run validate \
  --checkpoint runs/ablations/no_context_binding/seed0/best.pt

# Retrieve validation scores only, then evaluate the exported rankings.
python -m rcr_ablations.run retrieve \
  --checkpoint runs/ablations/no_context_binding/seed0/best.pt --split val
python -m rcr_ablations.run evaluate \
  --config runs/ablations/no_context_binding/seed0/config.yaml --split val

# Optional offline recalibration from a complete validation artifact.
python -m rcr_ablations.run calibrate \
  --rankings runs/ablations/no_context_binding/seed0/val/rankings.pt \
  --output-dir runs/ablations/no_context_binding/seed0/recalibrated
```

Offline calibration uses the original coefficient grid and can also accept an original calibration YAML with `--config`. It needs no model, backbone, checkpoint or GPU. It verifies the complete validation split and the artifact's original scoring contract.

After all validation choices are frozen, the following is the final test protocol; **do not use it during implementation, debugging or hyperparameter selection**:

```bash
python -m rcr_ablations.run run \
  --config runs/ablations/no_context_binding/seed0/selected.yaml --split test
```

`validate` and `run` delegate to the original runner, which retrieves and evaluates in separate processes. Full-mAP, Full-Hit@1, ID-mAP and Macro-Full-mAP retain the original definitions. The test guard still requires the checkpoint-bound validation selection.

Saved complete configurations also work with the original commands:

```bash
python third_party/rcr/tools/run.py retrieve \
  --config runs/ablations/no_context_binding/seed0/selected.yaml --split val
```

## Configuration and paths

`--config` accepts either an overlay or a saved full YAML. `--set KEY=VALUE` uses the original known-key override parser. Inspect a configuration without training:

```bash
python -m rcr_ablations.run train --variant no_both_bindings --seed 2 --print-config
```

Relative dataset/frozen-feature-cache/native-encoder paths resolve against `third_party/rcr`; relative output/checkpoint and derived supervision-cache paths resolve against this repository. Absolute paths are retained. CLI `--config`, `--checkpoint` and explicit `--resume` filenames resolve from the caller's directory. Commands work from any directory after editable installation. Change data/cache paths with `--set` to reuse assets elsewhere.

Native FAFA YAML path fields are resolved into an immutable, content-named file under `configs/generated/`, outside Proposed. The authoritative YAML is unchanged. Its original source commit, checkpoint, preprocessing and text policy are preserved. Cache-only machines may omit the native YAML, as supported by Proposed.

## CPU verification

```bash
python -m pytest -q
cd third_party/rcr
python -m pytest -q
```

Install the editable packages first so subprocess tests also resolve both packages. Without installation, set `PYTHONPATH` to the absolute `src`, `third_party/rcr/src` and `third_party/rcr` directories.

Tests use synthetic features and the upstream tiny cache/text doubles. They cover binding invocation isolation, original matching/null/coverage behavior, both FineReasoner attention rounds, masks and empty people, finite losses and gradients, training/inference consistency, strict checkpoint round trips, legacy B0 loading, metadata selection and exact interrupted/resumed CPU training. B0 is also compared bitwise in train and eval mode with an archived, untouched source tree at the original commit, including initialization, scores and gradients. No alternate RCR implementation is stored here.

See [implementation details](docs/implementation.md) for the execution paths, shared parameters, checkpoint contract and future extension points. No benchmark metrics are claimed by this integration.

The checked revisions, CPU environment and verification results are recorded in [verification](docs/verification.md).
