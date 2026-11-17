"""Small YAML overlays over Proposed, with paths independent of the working dir."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import yaml

from rcr.common.config import load_config, override_config, validate_training_schedule
from rcr_ablations.factory import MODEL_CLASSES, configure_variant

ROOT = Path(__file__).resolve().parents[2]
RCR_ROOT = ROOT / "third_party/rcr"
UPSTREAM_COMMIT = "88db1f011c21cfb6b3c76f32d72fb7a5646df673"


def _merge(base: dict, overlay: dict) -> dict:
    result = deepcopy(base)
    for key, value in overlay.items():
        result[key] = (
            _merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else deepcopy(value)
        )
    return result


def _absolute(value: str, root: Path) -> str:
    path = Path(value).expanduser()
    return str((root / path).resolve())


def resolve_paths(cfg: dict, *, root: Path = ROOT, rcr_root: Path = RCR_ROOT) -> dict:
    cfg = deepcopy(cfg)
    for key in (
        "final_dir",
        "image_root",
        "detection_cache",
        "scene_cache",
        "fafa_image_cache",
        "fafa_query_cache",
    ):
        cfg["data"][key] = _absolute(cfg["data"][key], rcr_root)
    cache = cfg.get("cache", {})
    if cache.get("supervision_cache"):
        cache["supervision_cache"] = _absolute(cache["supervision_cache"], root)
    person = cfg.get("person_encoder", {})
    if person.get("fafa_config"):
        person["fafa_config"] = _absolute(person["fafa_config"], rcr_root)
    if "/" in person.get("python", ""):
        person["python"] = _absolute(person["python"], rcr_root)
    cfg["output"]["dir"] = _absolute(cfg["output"]["dir"], root)
    cfg["checkpoint"] = _absolute(cfg["checkpoint"], root)
    return cfg


def materialize_native_config(cfg: dict, *, root: Path = ROOT) -> dict:
    """Resolve native FAFA file paths without editing its authoritative YAML."""
    cfg = deepcopy(cfg)
    path = Path(cfg.get("person_encoder", {}).get("fafa_config", ""))
    if not path.is_file():
        return cfg  # Upstream also permits cache-only use without the native YAML.
    native = load_config(path)
    for section, keys in (
        ("source", ("local_checkout",)),
        ("checkpoint", ("path", "cache_root", "runtime_assets_marker", "local_path")),
    ):
        for key in keys:
            if native.get(section, {}).get(key):
                native[section][key] = _absolute(native[section][key], RCR_ROOT)
    content = yaml.safe_dump(native, sort_keys=False)
    digest = sha256(content.encode()).hexdigest()[:16]
    destination = root / "configs/generated" / f"person_encoder-{digest}.yaml"
    from rcr.common.io import atomic_write_text

    if not destination.exists():
        atomic_write_text(destination, content)
    cfg["person_encoder"]["fafa_config"] = str(destination)
    return cfg


def load_effective_config(
    path: str | Path | None = None,
    *,
    variant: str | None = None,
    seed: int | None = None,
    checkpoint: str | Path | None = None,
    overrides: dict | None = None,
) -> dict:
    """Accept either a tiny overlay or a saved, complete effective config."""
    base = load_config(RCR_ROOT / "configs/proposed.yaml")
    saved = None
    if checkpoint is not None:
        import torch

        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    overlay = (
        load_config(path)
        if path is not None
        else (deepcopy(saved["config"]) if saved is not None else {})
    )
    selected = variant or overlay.get(
        "variant", saved["config"].get("variant", "proposed") if saved else "proposed"
    )
    if selected not in MODEL_CLASSES:
        raise ValueError(f"unknown variant: {selected}")
    if overlay.get("variant", selected) != selected:
        raise ValueError("--variant disagrees with the configuration variant")
    if overlay.get("model_class", MODEL_CLASSES[selected]) != MODEL_CLASSES[selected]:
        raise ValueError("model_class disagrees with the selected ablation variant")
    if path is None and saved is None and selected != "proposed":
        overlay = load_config(ROOT / "configs/variants" / f"{selected}.yaml")
    cfg = configure_variant(_merge(base, overlay), selected)
    if seed is not None:
        cfg["train"]["seed"] = seed
    # These defaults are the same ones filled in by the original trainer.
    cfg.setdefault("checkpointing", {"every_steps": 50})
    cfg = override_config(cfg, overrides or {})
    if "output" not in overlay and "output.dir" not in (overrides or {}):
        cfg["output"]["dir"] = str(
            ROOT / "runs/ablations" / selected / f"seed{cfg['train']['seed']}"
        )
    if cfg["variant"] != selected or cfg["model_class"] != MODEL_CLASSES[selected]:
        raise ValueError("use --variant to select a model, not --set")
    if checkpoint is not None:
        cfg["checkpoint"] = str(Path(checkpoint).expanduser().resolve())
    elif "checkpoint" not in overlay and "checkpoint" not in (overrides or {}):
        cfg["checkpoint"] = str(Path(cfg["output"]["dir"]) / "best.pt")
    cfg = resolve_paths(cfg)
    if cfg.get("method") != "proposed":
        raise ValueError("method must remain proposed")
    if cfg["retrieval"]["top_m"] != 500 or cfg["candidate_ks"] != [500]:
        raise ValueError("these ablations preserve top_m=500 and candidate_ks=[500]")
    validate_training_schedule(cfg)
    if saved is not None:
        from rcr.proposed.factory import validate_checkpoint_model

        validate_checkpoint_model(cfg, saved)
    return cfg
