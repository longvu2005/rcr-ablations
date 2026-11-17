# CPU verification

Verified locally on 2026-11-18:

| Check | Result |
| --- | --- |
| Ablation CPU suite | 81 passed |
| Unmodified Proposed test suite against the integration | 97 passed |
| Ruff checks for ablation sources/tests and the five Proposed integration files | Passed |
| Git whitespace checks in both repositories | Passed |
| Module CLI/import/config smoke from `/private/tmp`, all four variants, seed 5 | Passed |
| B0 versus untouched original source, train and eval modes | Bitwise equal state, RNG, query encodings, scores and gradients |
| B2/B3 direct evidence dtype, FP32 and CPU bfloat16 autocast | Matches original binding's FP32 output boundary |

The Proposed integration commit is `cd4731b46ca3faff12638c2359ed75e7955b180e`, on `feature/evidence-binding-ablations`. Proposed `main` remains `88db1f011c21cfb6b3c76f32d72fb7a5646df673`. All ablation-specific tests are outside Proposed.

The local CPU developer environment used Python 3.11.13, PyTorch 2.2.2, NumPy 1.26.4, SciPy 1.17.1, Transformers 4.57.6 and pytest 9.1.1. The CPU tests load no pretrained weights. Real experiments should use the original Proposed/FAFA dependency environments documented upstream; the integration does not change those requirements.

From this checkout, the source-only equivalent of the editable-install test commands is:

```bash
export PYTHONPATH="$PWD/src:$PWD/third_party/rcr/src:$PWD/third_party/rcr"
python -m pytest -q
cd third_party/rcr
python -m pytest -q
```

Full optimizer/resume and retrieval tests use only the original tiny synthetic cache and text doubles, with a two-candidate shortlist. Configuration tests separately assert the real experiments retain `top_m=500`, the original coefficient grids and all numerical training settings. The original dataset's test split was not evaluated. No full GPU training, benchmark metrics, merge or push was performed.
