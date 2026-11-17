"""Experiment fingerprints supplement the original cache/runtime provenance."""

import hashlib
import json
import subprocess
from pathlib import Path

from rcr_ablations.config import RCR_ROOT, ROOT, UPSTREAM_COMMIT
from rcr_ablations.factory import REPLACEMENTS, VARIANT_IDS


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sources(root: Path, paths) -> dict:
    files = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
        if not path.name.startswith(".")
    }
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def experiment_metadata(cfg: dict) -> dict:
    variant = cfg["variant"]
    ablation_files = [
        *(ROOT / "src/rcr_ablations").rglob("*.py"),
        *(ROOT / "configs/variants").glob("*.yaml"),
        ROOT / "pyproject.toml",
    ]
    proposed_files = [
        *(RCR_ROOT / "src/rcr").rglob("*.py"),
        RCR_ROOT / "configs/proposed.yaml",
        RCR_ROOT / "configs/person_encoder.yaml",
        RCR_ROOT / "pyproject.toml",
    ]
    return {
        "schema_version": 1,
        "variant_id": VARIANT_IDS[variant],
        "variant": variant,
        "model_class": cfg["model_class"],
        "replacement_components": REPLACEMENTS[variant],
        "upstream_original_commit": UPSTREAM_COMMIT,
        "proposed": {
            "commit": _git(RCR_ROOT, "rev-parse", "HEAD"),
            "source": _sources(RCR_ROOT, proposed_files),
        },
        "ablations": {
            "commit": _git(ROOT, "rev-parse", "HEAD"),
            "source": _sources(ROOT, ablation_files),
        },
    }


def write_run_manifest(output: Path) -> None:
    """Read the trainer's committed checkpoint, including its exact resume inputs."""
    import torch

    from rcr.common.io import sha256_file, write_json
    from rcr.evaluation.provenance import digest_json

    checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    state = checkpoint["training_state"]
    write_json(
        output / "manifest.json",
        {
            "model_metadata": checkpoint["model_metadata"],
            "effective_config": checkpoint["config"],
            "effective_config_sha256": digest_json(checkpoint["config"]),
            "initial_config_file_sha256": sha256_file(output / "config.yaml"),
            "resume_signature": state["signature"],
            "progress": {
                key: state["progress"][key] for key in ("epoch", "phase", "global_step")
            },
            "checkpoints": {
                name: sha256_file(output / name)
                for name in ("last.pt", "best.pt")
                if (output / name).is_file()
            },
        },
    )
