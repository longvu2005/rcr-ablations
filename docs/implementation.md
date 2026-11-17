# Implementation and scientific isolation

The original source of truth is Proposed commit `88db1f011c21cfb6b3c76f32d72fb7a5646df673`. The submodule gitlink pins the integration commit containing only shared extension points and checkpoint compatibility. Its exact SHA is recorded by Git and in every training manifest.

## Actual paths and shared parameters

The inspected implementation follows:

```text
encode_query
  -> shared person_proj / identity_head
  -> SubjectGrounding(scene, semantic persons, boxes, selection tokens)
  -> shared EvidenceBinding -> grounding.score -> logits [B,S,K]
  -> detached exclusive_membership (Subject/background competition and deduplication)
  -> StructuredComposition -> QueryEncoding

encode_grounded_identity
  -> shared person_proj -> the same self.grounding -> identity_head

score_context
  -> shared person_proj / detached identity_head
  -> partial_identity_matching (global one-to-one assignment with private nulls)
  -> gather_matched -> aligned persons/boxes in [B,S*K,D]
  -> bind_target_evidence
  -> mask unmatched evidence -> TargetPersonBuilder
  -> null_person substitution -> FineReasoner (two geometry-aware attention rounds)
  -> original FAFA / coverage / linear fine score and coarse fusion
```

In original RCR, there are **not two separately parameterized EvidenceBinding modules**: `SubjectGrounding.binding` owns one module and `score_context` calls that same object. Grounding and context also share `person_proj` and `visual_proj`; identity uses the same person semantics through `identity_head`. The ablations retain these names and the original object for every path that still uses it.

## B1 replacement

`TextPersonGrounding` receives the original grounding signature. For each Subject independently, it averages the unmasked selection tokens. A fully masked or zero-length selection has a zero pooled vector. For each Subject/person pair it concatenates `[person, pooled_text, person * pooled_text]`, then applies `Linear(3D, max(1,D//4)) -> GELU -> Linear(...,1)`. The logits have the original `[B,S,K]` shape. There is no normalization or competition across Subjects or people inside the scorer.

It uses no scene patches, boxes, target features, identity labels or conditions. Selection text and the original person semantics remain trainable. Grounding labels, the balanced grounding loss, exclusive/background membership and downstream modules are unchanged. Both `encode_query` and the coarse/warmup `encode_grounded_identity` call this replacement through the inherited `self.grounding` interface.

`NoGroundingBinding` constructs the entire original model first and replaces only `self.grounding`. The replacement holds the **original** `grounding.binding` object for context, with identical initial weights and parameter names. Head initialization runs under a CPU RNG fork, so all unchanged modules and the subsequent training RNG start identically. The original one-layer grounding score head is replaced by the MLP. Context continues to use the original EvidenceBinding attention, geometry, FFN, dropout and normalizations.

## B2 replacement

`NoContextBinding.bind_target_evidence` returns the supplied aligned semantic persons directly, casting them to FP32 to match the original binding's FP32 attention/residual output boundary under AMP. It reads no target scene patches and invokes neither EvidenceBinding attention operation. The returned tensor has the original `[B,S*K,D]` shape, device and output dtype. This cast adds no learned transform or scene evidence.

The inherited `score_context` still does identity matching, alignment, unmatched masking, target building, null substitution, relation geometry, membership priors and both FineReasoner attention rounds. Composed query tokens remain available to FineReasoner. Grounding retains the original EvidenceBinding. FAFA, coverage and all scoring/selection/calibration logic remain untouched.

## B3 reuse

`NoBothBindings(NoGroundingBinding, NoContextBinding)` inherits B1's constructor and B2's hook. It introduces no separate grounding or context algorithm. The original binding remains registered for initialization/state-layout consistency but is never called, receives no gradients and acquires no optimizer state. Its inactive presence is not evidence used by B3. The original grounding linear head is absent, as in B1.

## Factory and checkpoint contract

Proposed's small `factory.py` supports an optional explicit `model_class: module:Class` and defaults to the original `RCRModel`. Training and retrieval both call it. The experiment's closed mapping in `rcr_ablations.factory` selects exactly B0–B3. Ordinary Python imports select subclasses; no global registration, monkey-patching, rewritten source or copied pipeline is involved.

Checkpoints keep the complete effective `config`, original architecture version and original encoder/cache/training state. They additionally store `model_metadata` containing the variant name, class and experiment record. `experiment` records the B0–B3 ID, exact replacement components, original upstream SHA, checked-out Proposed SHA, ablation SHA and per-file source SHA256 maps. The fingerprints describe local edits even before a commit is published.

Retrieval validates metadata against the saved configuration and any explicit requested model before constructing the saved class. `composition.role.weight` still exists in every variant, so the original max-Subject reconstruction works. Strict state loading remains enabled. A metadata-free checkpoint is accepted only as an original Proposed checkpoint. An ablation cannot be loaded or resumed as a different variant even when two state layouts happen to match.

The original resume signature still hashes the whole RCR package, configuration, dataset inputs, caches and runtime. Optional `source_packages: [rcr_ablations]` additionally hashes every runtime Python source file in the ablation package **from disk**, rather than trusting a stale saved fingerprint. Numerical/source mismatches reject resume before training mutates the run's artifacts.

Each CLI training run writes `config.yaml` through the original trainer, plus `manifest.json` containing the latest checkpoint configuration, initial config-file hash, model metadata, complete resume signature, committed progress and checkpoint hashes. The signature supplies the original input/cache and runtime fingerprints. `selected.yaml` and ranking provenance retain the original validation-bound selection contract. A resumed run's original `config.yaml` stays as written by Proposed; the checkpoint and manifest contain the current effective configuration.

## Shared Proposed changes

Only five runtime files change in Proposed:

| File | Change |
| --- | --- |
| `proposed/factory.py` | Default model construction, explicit subclass selection, metadata validation and optional source-package hashing |
| `proposed/nn/model.py` | Move the original target binding call into `bind_target_evidence`; default operation is unchanged |
| `proposed/train.py` | Use the factory; save model metadata; validate the requested model on resume |
| `proposed/retrieve.py` | Validate metadata and reconstruct through the same factory using checkpoint-authoritative settings |
| `proposed/resume.py` | Include optional experiment-package hashes in the existing signature |

There are no changes to Proposed configs, losses, matching, composition, reasoner, encoders, annotations, datasets, ranking, calibration, evaluation definitions or training schedules. Tests for these integrations live in this repository; the original Proposed tests are also run.

Adding another ablation later normally requires a small subclass/replacement, a factory mapping and a YAML overlay. Hooks should be added to Proposed only when an existing small override cannot express the experiment.
