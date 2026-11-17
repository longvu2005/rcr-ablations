"""B2: feed matched person semantics directly to the original TargetPersonBuilder."""

from torch import Tensor

from rcr.proposed.nn.model import RCRModel


class NoContextBinding(RCRModel):
    def bind_target_evidence(
        self,
        scene: Tensor,
        persons: Tensor,
        boxes: Tensor,
        reference: Tensor,
        patch_hw: tuple[int, int],
        reference_mask: Tensor | None = None,
        reference_key_bias: Tensor | None = None,
    ) -> Tensor:
        # Original EvidenceBinding's FP32 person-attention residual promotes its
        # output to FP32 even under AMP. Preserve that boundary without binding.
        return persons.float()
