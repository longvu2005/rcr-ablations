"""Exercise the real trainer, resumable state, validation, calibration and retrieval."""

import copy
import json
import signal
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from tests.helpers import TextBackbone, Tokenizer
from tests.test_resume import _equal

from rcr.common.config import load_config
from rcr.common.io import sha256_file
from rcr.proposed import retrieve as retrieval
from rcr.proposed import train as training
from rcr.proposed.factory import (
    create_model,
    model_metadata,
    source_fingerprints,
    validate_checkpoint_model,
)
from rcr.proposed.nn.encoders import TextEncoder
from rcr.proposed.resume import TrainingProgress, resume_signature, verify_resume
from rcr_ablations.factory import MODEL_CLASSES, configure_variant
from rcr_ablations.provenance import experiment_metadata, write_run_manifest

VARIANTS = tuple(MODEL_CLASSES)


def test_training_cli_writes_complete_run_manifest(pipeline_run, monkeypatch):
    from rcr_ablations import run

    cfg, _, _, _ = pipeline_run
    cfg = configure_variant(cfg, "no_both_bindings")
    monkeypatch.setattr(run, "load_effective_config", lambda *args, **kwargs: cfg)
    result = run.main(["train", "--variant", "no_both_bindings"])
    manifest = json.loads((result.parent / "manifest.json").read_text())
    assert manifest["model_metadata"]["experiment"]["variant_id"] == "B3"
    assert manifest["model_metadata"]["experiment"]["replacement_components"] == {
        "grounding": "masked_mean_text_person_mlp",
        "context": "direct_identity_aligned_person_features",
    }
    assert manifest["effective_config"]["train"]["seed"] == cfg["train"]["seed"]
    assert "experiment" in load_config(result.parent / "config.yaml")
    digest = sha256_file(result.parent / "last.pt")
    assert run.main(["train", "--variant", "no_both_bindings", "--resume"]) == result
    assert sha256_file(result.parent / "last.pt") == digest


@pytest.mark.parametrize("variant", VARIANTS)
def test_train_resume_strict_load_validate_and_retrieve(
    variant, pipeline_run, monkeypatch
):
    cfg, cache, _, _ = pipeline_run
    cfg = configure_variant(cfg, variant)
    cfg["train"].update(epochs=2, context_warmup_epochs=1)
    cfg["evaluation"]["calibration"] = {
        "coverage_weights": [0.04],
        "context_weights": [0.1],
        "fine_coarse_weights": [0.15],
    }
    cfg["experiment"] = experiment_metadata(cfg)
    reference_cfg = copy.deepcopy(cfg)
    reference_cfg["output"]["dir"] += "-reference"
    reference = training.train(reference_cfg)
    expected = torch.load(reference.parent / "last.pt", weights_only=True)
    compute = training.compute_loss

    def pause(*args, **kwargs):
        result = compute(*args, **kwargs)
        signal.raise_signal(signal.SIGTERM)
        return result

    with monkeypatch.context() as interrupted:
        interrupted.setattr(training, "compute_loss", pause)
        training.train(cfg)
    root = Path(cfg["output"]["dir"])
    pending = torch.load(root / "last.pt", weights_only=True)
    assert pending["training_state"]["progress"]["global_step"] == 1
    result = training.train(cfg, resume="auto")
    actual = torch.load(root / "last.pt", weights_only=True)
    for key in ("model", "text_encoder", "best_epoch", "best_score"):
        _equal(actual[key], expected[key])
    for key in ("optimizer", "scaler", "rng"):
        _equal(actual["training_state"][key], expected["training_state"][key])
    assert (root / "history.jsonl").read_text() == (
        reference.parent / "history.jsonl"
    ).read_text()
    assert actual["model_metadata"]["variant"] == variant
    assert actual["model_metadata"]["model_class"] == MODEL_CLASSES[variant]
    assert (
        "rcr_ablations"
        in actual["training_state"]["signature"]["source_packages_sha256"]
    )
    assert actual["model_metadata"]["experiment"]["proposed"]["commit"]
    assert actual["model_metadata"]["experiment"]["ablations"]["source"]["sha256"]
    restored = create_model(
        actual["config"],
        input_dim=actual["input_dim"],
        person_input_dim=actual["person_input_dim"],
        global_dim=actual["global_dim"],
        max_subjects=actual["model"]["composition.role.weight"].shape[0],
    )
    restored.load_state_dict(actual["model"], strict=True)
    invalid = dict(actual["model"])
    invalid.pop("composition.role.weight")
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(invalid, strict=True)
    assert result.name == "best.pt"
    write_run_manifest(root)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["model_metadata"] == actual["model_metadata"]
    assert manifest["checkpoints"]["last.pt"] == sha256_file(root / "last.pt")
    selected = load_config(root / "selected.yaml")
    assert selected["variant"] == variant
    assert selected["selected_retrieval"]["top_m"] == 2  # Tiny synthetic gallery only.
    data = training.load_rcr_data(None)
    monkeypatch.setattr(retrieval, "load_rcr_data", lambda *args: data)
    monkeypatch.setattr(retrieval, "GalleryCache", lambda **kwargs: cache)
    monkeypatch.setattr(retrieval, "check_cache_config", lambda *args: None)
    monkeypatch.setattr(retrieval, "check_query_text_config", lambda *args: None)
    monkeypatch.setattr(
        retrieval,
        "load_text_modules",
        lambda *args, **kwargs: (Tokenizer(), TextEncoder(TextBackbone(), dim=6)),
    )
    output = retrieval.retrieve(selected)
    assert output["sample_ids"] == ["val"]
    assert output["provenance"]["config"]["variant"] == variant
    assert torch.isfinite(output["context"]).all()
    digest = sha256_file(root / "last.pt")
    wrong = configure_variant(
        cfg, "no_both_bindings" if variant != "no_both_bindings" else "proposed"
    )
    with pytest.raises(ValueError, match="checkpoint variant"):
        training.train(wrong, resume="auto")
    assert sha256_file(root / "last.pt") == digest
    with pytest.raises(FileExistsError, match="already exists"):
        training.train(cfg)


@pytest.mark.parametrize("saved_variant", VARIANTS)
@pytest.mark.parametrize("requested_variant", VARIANTS)
def test_saved_metadata_selects_variant_and_rejects_mismatch(
    saved_variant, requested_variant
):
    cfg = configure_variant({"model": {}}, saved_variant)
    checkpoint = {"config": cfg, "model_metadata": model_metadata(cfg)}
    request = configure_variant({"model": {}}, requested_variant)
    if saved_variant == requested_variant:
        validate_checkpoint_model(request, checkpoint)
    else:
        with pytest.raises(ValueError, match="checkpoint variant"):
            validate_checkpoint_model(request, checkpoint)
    validate_checkpoint_model({}, checkpoint)
    checkpoint["model_metadata"] = {**checkpoint["model_metadata"], "variant": "broken"}
    with pytest.raises(ValueError, match="metadata disagrees"):
        validate_checkpoint_model({}, checkpoint)


def test_legacy_proposed_checkpoint_still_retrieves(saved_pipeline):
    cfg, _, _ = saved_pipeline
    path = Path(cfg["checkpoint"])
    checkpoint = torch.load(path, weights_only=True)
    checkpoint.pop("model_metadata")
    torch.save(checkpoint, path)
    output = retrieval.retrieve(cfg)
    assert output["provenance"]["method"] == "proposed"


def test_resume_hashes_source_files_from_disk(pipeline_run, tmp_path, monkeypatch):
    cfg, cache, _, _ = pipeline_run
    cfg = configure_variant(cfg, "no_both_bindings")
    assert "rcr_ablations" in source_fingerprints(cfg)
    package = tmp_path / "experiment_integrity_test"
    package.mkdir()
    (package / "__init__.py").write_text("")
    source = package / "replacement.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    cfg["source_packages"] = ["experiment_integrity_test"]
    data = training.load_rcr_data(None)
    signature = resume_signature(cfg, data, cache, torch.device("cpu"), False)
    checkpoint = {
        "training_state": {
            "signature": signature,
            "progress": asdict(TrainingProgress()),
        }
    }
    verify_resume(checkpoint, signature)
    source.write_text("VALUE = 2\n")
    current = resume_signature(cfg, data, cache, torch.device("cpu"), False)
    with pytest.raises(ValueError, match="source_packages_sha256"):
        verify_resume(checkpoint, current)
