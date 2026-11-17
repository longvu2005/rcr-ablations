"""The four supported model selections; no runtime registration or patching."""

from copy import deepcopy

MODEL_CLASSES = {
    "proposed": "rcr.proposed.nn.model:RCRModel",
    "no_grounding_binding": (
        "rcr_ablations.variants.no_grounding_binding:NoGroundingBinding"
    ),
    "no_context_binding": "rcr_ablations.variants.no_context_binding:NoContextBinding",
    "no_both_bindings": "rcr_ablations.variants.no_both_bindings:NoBothBindings",
}
VARIANT_IDS = dict(zip(MODEL_CLASSES, ("B0", "B1", "B2", "B3"), strict=True))
REPLACEMENTS = {
    "proposed": {},
    "no_grounding_binding": {"grounding": "masked_mean_text_person_mlp"},
    "no_context_binding": {"context": "direct_identity_aligned_person_features"},
    "no_both_bindings": {
        "grounding": "masked_mean_text_person_mlp",
        "context": "direct_identity_aligned_person_features",
    },
}


def configure_variant(cfg: dict, variant: str) -> dict:
    cfg = deepcopy(cfg)
    cfg.update(
        variant=variant,
        model_class=MODEL_CLASSES[variant],
        source_packages=["rcr_ablations"],
    )
    return cfg


def create_model(cfg: dict, **dimensions):
    from rcr.proposed.factory import create_model as create_rcr_model

    variant = cfg.get("variant", "proposed")
    if cfg.get("model_class", MODEL_CLASSES[variant]) != MODEL_CLASSES[variant]:
        raise ValueError("model_class disagrees with the selected ablation variant")
    return create_rcr_model(configure_variant(cfg, variant), **dimensions)
