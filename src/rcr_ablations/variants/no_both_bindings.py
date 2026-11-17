"""B3 is exactly the B1 grounding replacement and B2 context replacement."""

from rcr_ablations.variants.no_context_binding import NoContextBinding
from rcr_ablations.variants.no_grounding_binding import NoGroundingBinding


class NoBothBindings(NoGroundingBinding, NoContextBinding):
    pass
