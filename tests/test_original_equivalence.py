"""Compare B0 with an unmodified checkout of the exact original Proposed commit."""

import os
import subprocess
import sys
import tarfile

import pytest
import torch

from rcr_ablations.config import RCR_ROOT, ROOT, UPSTREAM_COMMIT

SCRIPT = r"""
import sys
import torch
from rcr.proposed.nn.model import RCRModel
torch.set_num_threads(1)
torch.manual_seed(712)
config = {"model": {"dim": 8, "identity_dim": 6, "num_heads": 2, "dropout": 0.1,
                    "fine_matching": {"background_logit": -3.5}}}
if sys.argv[1] == "factory":
    from rcr_ablations.factory import create_model
    model = create_model(config, input_dim=12, person_input_dim=10)
else:
    model = RCRModel.from_config(config["model"], input_dim=12, person_input_dim=10)
model.train(sys.argv[3] == "train")
rng = torch.get_rng_state()
persons = torch.randn(2, 3, 10)
boxes = torch.tensor([[0.,0.,.25,1.],[.35,0.,.6,1.],[.7,0.,1.,1.]]).expand(2,-1,-1)
query = model.encode_query(torch.randn(2,4,12), persons, boxes,
    torch.randn(2,2,4,8), torch.randn(2,5,8), torch.tensor([[1,3],[1,3]]), (2,2),
    selection_mask=torch.ones(2,2,4,dtype=torch.bool), fafa_query=torch.randn(2,2,3,5))
score = model.score_target(query, torch.randn(2,4,12), persons, boxes, (2,2),
    fafa_targets=torch.randn(2,3,3,5), return_details=True)
(query.logits.square().sum() + score["score"].sum()).backward()
torch.save({"state": model.state_dict(), "rng": rng,
    "query": {key: value.detach() for key, value in vars(query).items()},
    "scores": score,
    "gradients": {key: value.grad for key, value in model.named_parameters()}},
    sys.argv[2])
"""


def _equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _equal(actual[key], expected[key])
    elif isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert actual == expected


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_b0_is_bitwise_equivalent_to_unchanged_proposed(tmp_path, mode):
    archive = tmp_path / "original.tar"
    with archive.open("wb") as output:
        subprocess.run(
            ["git", "archive", UPSTREAM_COMMIT, "src/rcr"],
            cwd=RCR_ROOT,
            stdout=output,
            stderr=subprocess.PIPE,
            check=True,
        )
    original = tmp_path / "original"
    with tarfile.open(archive) as source:
        source.extractall(original, filter="data")
    script = tmp_path / "compare.py"
    script.write_text(SCRIPT)
    for name, pythonpath in (
        ("original", str(original / "src")),
        ("factory", os.pathsep.join((str(ROOT / "src"), str(RCR_ROOT / "src")))),
    ):
        subprocess.run(
            [sys.executable, str(script), name, str(tmp_path / f"{name}.pt"), mode],
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": pythonpath},
            capture_output=True,
            text=True,
            check=True,
        )
    _equal(
        torch.load(tmp_path / "factory.pt", weights_only=True),
        torch.load(tmp_path / "original.pt", weights_only=True),
    )
