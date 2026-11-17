"""Synthetic CPU inputs and reuse of Proposed's existing workflow doubles."""

import pytest
import torch

from rcr_ablations.config import RCR_ROOT
from tests.conftest import objective_batch as upstream_objective_batch
from tests.conftest import saved_pipeline as upstream_saved_pipeline
from tests.conftest import tiny_training_run as upstream_training_run


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def pipeline_run(tmp_path, monkeypatch):
    # Upstream's fixture reads its original configs/proposed.yaml relative to cwd.
    monkeypatch.chdir(RCR_ROOT)
    return upstream_training_run.__wrapped__(tmp_path, monkeypatch)


@pytest.fixture
def objective_batch():
    return upstream_objective_batch.__wrapped__()


@pytest.fixture
def saved_pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(RCR_ROOT)
    return upstream_saved_pipeline.__wrapped__(tmp_path, monkeypatch)


@pytest.fixture
def features():
    g = torch.Generator().manual_seed(41)
    people = torch.randn(2, 3, 8, generator=g)
    boxes = (
        torch.tensor(
            [[0.0, 0.0, 0.25, 0.8], [0.35, 0.0, 0.6, 0.8], [0.7, 0.0, 1.0, 0.8]]
        )
        .expand(2, -1, -1)
        .clone()
    )
    selection_mask = torch.ones(2, 2, 4, dtype=torch.bool)
    selection_mask[..., -1] = False
    selection_mask[1, 1] = False
    return {
        "query_scene": torch.randn(2, 4, 8, generator=g),
        "query_persons": people,
        "query_boxes": boxes,
        "selections": torch.randn(2, 2, 4, 8, generator=g),
        "change": torch.randn(2, 5, 8, generator=g),
        "subject_pos": torch.tensor([[1, 3], [1, 3]]),
        "patch_hw": (2, 2),
        "selection_mask": selection_mask,
        "change_mask": torch.ones(2, 5, dtype=torch.bool),
        "query_person_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "subject_mask": torch.tensor([[True, True], [True, False]]),
        "target_scene": torch.randn(2, 4, 8, generator=g),
        "target_persons": people.clone(),
        "target_boxes": boxes.clone(),
        "target_mask": torch.tensor([[True, True, True], [True, False, False]]),
        "fafa_query": torch.randn(2, 2, 3, 5, generator=g),
        "fafa_targets": torch.randn(2, 3, 3, 5, generator=g),
    }
