"""Prove binding isolation, unchanged matching/reasoning, and numerical contracts."""

from collections import Counter
from dataclasses import fields

import pytest
import torch

from rcr.proposed.losses import _context_scores, compute_loss, loss_settings
from rcr.proposed.matching import gather_matched, partial_identity_matching
from rcr.proposed.nn.binding import EvidenceBinding
from rcr.proposed.nn.composition import StructuredComposition
from rcr.proposed.nn.grounding import SubjectGrounding
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.nn.reasoning import FineReasoner, TargetPersonBuilder
from rcr_ablations.factory import MODEL_CLASSES, configure_variant, create_model
from rcr_ablations.variants.no_both_bindings import NoBothBindings
from rcr_ablations.variants.no_context_binding import NoContextBinding
from rcr_ablations.variants.no_grounding_binding import (
    NoGroundingBinding,
    TextPersonGrounding,
)

VARIANTS = tuple(MODEL_CLASSES)
QUERY_KEYS = (
    "query_scene",
    "query_persons",
    "query_boxes",
    "selections",
    "change",
    "subject_pos",
    "patch_hw",
    "selection_mask",
    "change_mask",
    "query_person_mask",
    "subject_mask",
    "subject_token_mask",
    "subject_ids",
    "fafa_query",
)


def make_model(variant, **settings):
    torch.manual_seed(7)
    cfg = configure_variant(
        {"model": {"dim": 8, "identity_dim": 6, "num_heads": 2, **settings}},
        variant,
    )
    return create_model(cfg, input_dim=8, person_input_dim=8).eval()


def encode(model, features):
    return model.encode_query(
        **{key: features[key] for key in QUERY_KEYS if key in features}
    )


def context(model, query, features):
    return model.score_context(
        query,
        features["target_scene"],
        features["target_persons"],
        features["target_boxes"],
        features["patch_hw"],
        features["target_mask"],
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_only_intended_modules_and_common_initialization_change(variant):
    reference = make_model("proposed")
    reference_rng = torch.get_rng_state()
    model = make_model(variant)
    assert torch.equal(torch.get_rng_state(), reference_rng)
    assert type(model.composition) is StructuredComposition
    assert type(model.target_builder) is TargetPersonBuilder
    assert type(model.reasoner) is FineReasoner
    assert type(model.grounding.binding) is EvidenceBinding
    assert model.score_context.__func__ is RCRModel.score_context
    assert model.encode_grounded_identity.__func__ is RCRModel.encode_grounded_identity
    replacement = variant in ("no_grounding_binding", "no_both_bindings")
    assert type(model.grounding) is (
        TextPersonGrounding if replacement else SubjectGrounding
    )
    for key, value in reference.state_dict().items():
        if replacement and key.startswith("grounding.score."):
            continue
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    if variant == "proposed":
        assert type(model) is RCRModel


@pytest.mark.parametrize("variant", VARIANTS)
def test_grounding_and_context_binding_invocations(variant, features):
    model = make_model(variant, fine_matching={"background_logit": -3.5})
    calls = Counter()
    handles = []
    for name, module in (
        ("binding", model.grounding.binding),
        ("ref_attn", model.grounding.binding.ref_attn),
        ("person_attn", model.grounding.binding.person_attn),
        ("composition", model.composition),
        ("builder", model.target_builder),
        ("reasoner", model.reasoner),
        ("reasoner_attention", model.reasoner.self_attn),
        ("relation", model.reasoner.relation_geo),
    ):

        def count(_module, _args, _result, key=name):
            calls[key] += 1

        handles.append(module.register_forward_hook(count))
    query = encode(model, features)
    original_ground = variant in ("proposed", "no_context_binding")
    assert calls["binding"] == calls["ref_attn"] == int(original_ground)
    assert calls["person_attn"] == int(original_ground)
    people = features["query_persons"].masked_fill(
        ~features["query_person_mask"][..., None], 0
    )
    boxes = features["query_boxes"].masked_fill(
        ~features["query_person_mask"][..., None], 0
    )
    logits, identity = model.encode_grounded_identity(
        features["query_scene"],
        people,
        boxes,
        features["selections"],
        (2, 2),
        features["selection_mask"],
    )
    torch.testing.assert_close(logits, query.logits, rtol=0, atol=0)
    torch.testing.assert_close(identity, query.identity, rtol=0, atol=0)
    assert calls["binding"] == 2 * int(original_ground)
    calls.clear()
    scores, matching = context(model, query, features)
    assert matching.matched.any()
    original_context = variant in ("proposed", "no_grounding_binding")
    assert calls["binding"] == calls["ref_attn"] == int(original_context)
    assert calls["person_attn"] == int(original_context)
    assert calls["builder"] == calls["reasoner"] == calls["relation"] == 1
    assert calls["reasoner_attention"] == 2
    assert query.logits.shape == (2, 2, 3)
    assert query.reference.shape == (2, 12, 8)
    assert scores.shape == (2,) and torch.isfinite(scores).all()
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("variant", ["no_grounding_binding", "no_both_bindings"])
def test_text_conditioning_independent_subjects_masked_mean(variant, features):
    model = make_model(variant)
    original = encode(model, features).logits
    changed = dict(features)
    changed["selections"] = features["selections"].clone()
    changed["selections"][:, 0, :3] += 10
    new = encode(model, changed).logits
    assert not torch.allclose(original[:, 0], new[:, 0])
    torch.testing.assert_close(original[:, 1], new[:, 1], rtol=0, atol=0)
    changed["selections"] = features["selections"].masked_fill(
        ~features["selection_mask"][..., None], torch.nan
    )
    torch.testing.assert_close(encode(model, changed).logits, original, rtol=0, atol=0)
    changed = {**features, "query_scene": torch.full_like(features["query_scene"], 99)}
    torch.testing.assert_close(encode(model, changed).logits, original, rtol=0, atol=0)
    text = features["selections"].clone().requires_grad_()
    logits = encode(model, {**features, "selections": text}).logits
    logits.sum().backward()
    assert text.grad[features["selection_mask"]].abs().sum() > 0
    assert text.grad[~features["selection_mask"]].count_nonzero() == 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_alignment_membership_nulls_coverage_and_reasoner_preserved(variant, features):
    model = make_model(variant, fine_matching={"background_logit": -3.5})
    query = encode(model, features)
    captured = {}

    def builder(_module, args, result):
        captured.update(evidence=args[0], boxes=args[1], target=result)

    def reasoner(_module, args, kwargs):
        captured.update(reasoner_args=args, reasoner_kwargs=kwargs)

    model.target_builder.register_forward_hook(builder)
    model.reasoner.register_forward_pre_hook(reasoner, with_kwargs=True)
    _, matching = context(model, query, features)
    mask = features["target_mask"]
    persons = features["target_persons"].masked_fill(~mask[..., None], 0)
    boxes = features["target_boxes"].masked_fill(~mask[..., None], 0)
    semantic = model.encode_semantic(persons)
    expected = partial_identity_matching(
        query.identity,
        model.identity_head(semantic),
        query.membership,
        query.subjects,
        mask,
        boxes,
        identity_threshold=model.fine_matching["identity_threshold"],
        duplicate_iou=model.fine_matching["duplicate_iou"],
    )
    for field in fields(matching):
        torch.testing.assert_close(
            getattr(matching, field.name), getattr(expected, field.name), rtol=0, atol=0
        )
    b, s, k = matching.membership.shape
    selected = matching.membership > 0
    matched = selected & matching.matched[:, None]
    aligned = gather_matched(semantic, matching)[:, None].expand(-1, s, -1, -1)
    aligned = aligned.masked_fill(~matched[..., None], 0).reshape(b, s * k, 8)
    if variant in ("no_context_binding", "no_both_bindings"):
        torch.testing.assert_close(captured["evidence"], aligned, rtol=0, atol=0)
    torch.testing.assert_close(
        captured["reasoner_args"][1],
        torch.where(
            matched.flatten(1)[..., None], captured["target"], model.null_person
        ),
        rtol=0,
        atol=0,
    )
    assert torch.equal(captured["reasoner_args"][2], selected.flatten(1))
    assert torch.equal(captured["reasoner_kwargs"]["matched_mask"], matched.flatten(1))
    assert captured["reasoner_kwargs"]["aligned_offset"] == 6
    torch.testing.assert_close(
        captured["reasoner_kwargs"]["aligned_boxes"], captured["boxes"]
    )


@pytest.mark.parametrize("variant", ["no_context_binding", "no_both_bindings"])
def test_no_context_binding_is_independent_of_target_scene(variant, features):
    model = make_model(variant, fine_matching={"background_logit": -3.5})
    query = encode(model, features)
    original, _ = context(model, query, features)
    changed = {
        **features,
        "target_scene": torch.full_like(features["target_scene"], torch.nan),
    }
    new, _ = context(model, query, changed)
    torch.testing.assert_close(new, original, rtol=0, atol=0)


def test_b3_reuses_b1_and_b2_exactly(features):
    assert NoBothBindings.__init__ is NoGroundingBinding.__init__
    assert NoBothBindings.bind_target_evidence is NoContextBinding.bind_target_evidence
    b1, b3 = make_model("no_grounding_binding"), make_model("no_both_bindings")
    for name, value in b1.state_dict().items():
        torch.testing.assert_close(b3.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(encode(b1, features).logits, encode(b3, features).logits)


@pytest.mark.parametrize("variant", ["no_context_binding", "no_both_bindings"])
@pytest.mark.parametrize("amp", [False, True])
def test_direct_evidence_preserves_original_output_dtype(variant, amp, features):
    original = make_model("proposed", fine_matching={"background_logit": -3.5})
    replacement = make_model(variant, fine_matching={"background_logit": -3.5})
    evidence = []

    def capture(_module, args):
        evidence.append(args[0])

    original.target_builder.register_forward_pre_hook(capture)
    replacement.target_builder.register_forward_pre_hook(capture)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        query = encode(original, features)
        context(original, query, features)
        context(replacement, query, features)
    assert evidence[0].shape == evidence[1].shape == (2, 6, 8)
    assert evidence[0].dtype == evidence[1].dtype == torch.float32
    assert torch.isfinite(evidence[1]).all()


@pytest.mark.parametrize("variant", VARIANTS)
def test_actual_losses_backward_and_training_inference_consistency(
    variant, objective_batch
):
    _, batch = objective_batch
    model = make_model(
        variant,
        state_dim=4,
        fine_matching={"selection_threshold": 0.0001, "identity_threshold": -1.0},
    )
    total, losses = compute_loss(
        model, batch, (2, 2), **loss_settings({}), context_batch_size=1
    )
    assert total.ndim == 0 and torch.isfinite(total)
    assert all(
        torch.isfinite(torch.as_tensor(value)).all() for value in losses.values()
    )
    total.backward()
    modules = (
        model.grounding.score,
        model.person_proj,
        model.identity_head,
        model.composition,
        model.target_builder,
        model.reasoner,
        model.state_text_proj,
        model.state_image_proj,
    )
    for module in modules:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
    binding_grads = [p.grad for p in model.grounding.binding.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in binding_grads) == (
        variant != "no_both_bindings"
    )
    query = encode(model, batch | {"patch_hw": (2, 2)})
    raw, coverage, binding = _context_scores(model, query, batch, (2, 2), 1)
    details = model.score_target(
        query.repeat_candidates(3),
        batch["target_scene"].flatten(0, 1),
        batch["target_persons"].flatten(0, 1),
        batch["target_boxes"].flatten(0, 1),
        (2, 2),
        batch["target_mask"].flatten(0, 1),
        fafa_targets=batch["fafa_targets"].flatten(0, 1),
        return_details=True,
        pooling=batch["fafa_pooling"],
    )
    for name, train_values in (
        ("context", raw),
        ("coverage", coverage),
        ("binding", binding),
    ):
        torch.testing.assert_close(
            details[name], train_values.flatten(), rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "case", ["empty_query", "empty_target", "padding", "unmatched"]
)
def test_empty_padded_and_unmatched_slots(variant, case, features):
    data = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in features.items()
    }
    settings = {"background_logit": -3.5}
    if case == "empty_query":
        for key in ("query_persons", "query_boxes", "query_person_mask"):
            data[key] = data[key][:, :0]
        data["fafa_query"] = data["fafa_query"][:, :, :0]
    if case == "empty_target":
        for key in ("target_persons", "target_boxes", "target_mask", "fafa_targets"):
            data[key] = data[key][:, :0]
    if case == "padding":
        for prefix, mask_key in (
            ("query", "query_person_mask"),
            ("target", "target_mask"),
        ):
            data[mask_key].fill_(False)
            data[f"{prefix}_persons"].fill_(torch.nan)
            data[f"{prefix}_boxes"].fill_(torch.nan)
        data["selection_mask"].fill_(False)
        data["subject_mask"].fill_(False)
    if case == "unmatched":
        settings["identity_threshold"] = 1.0
    model = make_model(variant, fine_matching=settings)
    logits, score = model(**data)
    assert logits.shape == (2, 2, data["query_persons"].shape[1])
    assert score.shape == (2,) and torch.isfinite(score).all()
    assert torch.isfinite(logits).all()
    _, matching = context(model, encode(model, data), data)
    assert not matching.matched.any()
    assert matching.coverage.count_nonzero() == 0
