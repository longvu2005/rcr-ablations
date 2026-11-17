"""B1: selection-conditioned logits without grounding EvidenceBinding attention."""

import torch
from torch import Tensor, nn

from rcr.proposed.nn.binding import EvidenceBinding
from rcr.proposed.nn.model import RCRModel


class TextPersonGrounding(nn.Module):
    """Score each Subject/person pair; retain the original binding for context."""

    def __init__(self, binding: EvidenceBinding, dim: int) -> None:
        super().__init__()
        self.binding = binding
        self.score = nn.Sequential(
            nn.Linear(3 * dim, max(1, dim // 4)),
            nn.GELU(),
            nn.Linear(max(1, dim // 4), 1),
        )

    def forward(
        self,
        scene: Tensor,
        persons: Tensor,
        boxes: Tensor,
        selections: Tensor,
        patch_hw: tuple[int, int],
        selection_mask: Tensor | None = None,
    ) -> Tensor:
        """Selections [B,S,L,D], persons [B,K,D] -> independent logits [B,S,K]."""
        if selection_mask is None:
            pooled = selections.sum(-2) / max(1, selections.shape[-2])
        else:
            mask = selection_mask.bool()
            pooled = selections.masked_fill(~mask[..., None], 0).sum(-2)
            pooled = pooled / mask.sum(-1, keepdim=True).clamp_min(1)
        b, s, d = pooled.shape
        k = persons.shape[1]
        text = pooled[:, :, None].expand(b, s, k, d)
        person = persons[:, None].expand(b, s, k, d)
        return self.score(torch.cat((person, text, person * text), -1)).squeeze(-1)


class NoGroundingBinding(RCRModel):
    def __init__(self, *args, **kwargs) -> None:
        # Construct every original module first, preserving their initial values.
        super().__init__(*args, **kwargs)
        # Keep the subsequent training RNG trajectory unchanged by head creation.
        with torch.random.fork_rng(devices=[]):
            self.grounding = TextPersonGrounding(
                self.grounding.binding, self.person_proj[0].out_features
            )
