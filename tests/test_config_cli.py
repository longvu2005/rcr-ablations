"""Overlays, metadata-driven selection, path resolution and safe CLI defaults."""

from pathlib import Path

import pytest
import torch
import yaml

from rcr.common.config import load_config
from rcr_ablations.config import (
    RCR_ROOT,
    ROOT,
    load_effective_config,
    materialize_native_config,
)
from rcr_ablations.factory import MODEL_CLASSES, configure_variant
from rcr_ablations.run import main


@pytest.mark.parametrize("variant", tuple(MODEL_CLASSES))
@pytest.mark.parametrize("seed", [0, 3])
def test_overlay_preserves_original_experiment(variant, seed, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = load_config(RCR_ROOT / "configs/proposed.yaml")
    cfg = load_effective_config(variant=variant, seed=seed)
    assert cfg["method"] == "proposed"
    assert cfg["train"] == {**original["train"], "seed": seed}
    for key in (
        "model",
        "loss",
        "optimizer",
        "retrieval",
        "evaluation",
        "candidate_ks",
    ):
        assert cfg[key] == original[key]
    assert (
        Path(cfg["output"]["dir"]) == ROOT / "runs/ablations" / variant / f"seed{seed}"
    )
    assert Path(cfg["checkpoint"]) == Path(cfg["output"]["dir"]) / "best.pt"
    assert Path(cfg["data"]["final_dir"]) == RCR_ROOT / "dataset/data/final"
    assert all(Path(value).is_absolute() for value in cfg["data"].values())
    assert cfg["model_class"] == MODEL_CLASSES[variant]
    assert (
        Path(cfg["cache"]["supervision_cache"]) == ROOT / "cache/train-supervision.json"
    )


def test_effective_config_roundtrip_and_native_paths(tmp_path, monkeypatch):
    cfg = load_effective_config(variant="no_grounding_binding", seed=4)
    path = tmp_path / "effective.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    assert load_effective_config(path) == cfg
    resolved = materialize_native_config(cfg, root=tmp_path)
    native_path = Path(resolved["person_encoder"]["fafa_config"])
    assert native_path.is_relative_to(tmp_path)
    native = load_config(native_path)
    assert Path(native["source"]["local_checkout"]).is_absolute()
    for key in ("path", "cache_root", "runtime_assets_marker"):
        assert Path(native["checkpoint"][key]).is_absolute()
    assert materialize_native_config(resolved, root=tmp_path) == resolved


@pytest.mark.parametrize("variant", tuple(MODEL_CLASSES))
def test_checkpoint_drives_model_selection(variant, tmp_path):
    cfg = configure_variant(load_effective_config(), variant)
    path = tmp_path / "model.pt"
    from rcr.proposed.factory import model_metadata

    torch.save({"config": cfg, "model_metadata": model_metadata(cfg)}, path)
    loaded = load_effective_config(checkpoint=path)
    assert loaded["variant"] == variant
    assert loaded["model_class"] == MODEL_CLASSES[variant]
    assert loaded["checkpoint"] == str(path)
    wrong = "no_both_bindings" if variant != "no_both_bindings" else "proposed"
    with pytest.raises(ValueError, match="variant"):
        load_effective_config(checkpoint=path, variant=wrong)


def test_cli_prints_config_without_training_or_changing_cwd(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    main(["train", "--variant", "no_context_binding", "--seed", "2", "--print-config"])
    cfg = yaml.safe_load(capsys.readouterr().out)
    assert cfg["variant"] == "no_context_binding"
    assert cfg["train"]["seed"] == 2
    assert cfg["output"]["dir"].endswith("no_context_binding/seed2")
    assert Path.cwd() == tmp_path
    cfg = load_effective_config(overrides={"train.seed": 7})
    assert cfg["output"]["dir"].endswith("proposed/seed7")


def test_cli_rejects_changed_protocol_and_existing_runs(tmp_path):
    with pytest.raises(ValueError, match="top_m=500"):
        load_effective_config(overrides={"retrieval.top_m": 10})
    (tmp_path / "config.yaml").write_text("existing: run\n")
    with pytest.raises(SystemExit) as error:
        main(
            [
                "train",
                "--variant",
                "no_both_bindings",
                "--set",
                f"output.dir={tmp_path}",
            ]
        )
    assert error.value.code == 2
    assert (tmp_path / "config.yaml").read_text() == "existing: run\n"
