
from typing import NamedTuple
from dmalax.types import PyTree, PRNGKey

__all__ = ["MALAState", "MALAInfo", "SGLDState"]


class MALAState(NamedTuple):
    """State of the MALA algorithm.

    The MALA algorithm takes one position of the chain and returns another
    position. In order to make computations more efficient, we also store
    the current log-probability density as well as the current gradient of the
    log-probability density.

    """

    position: PyTree
    logprob: float
    logprob_grad: PyTree


class MALAInfo(NamedTuple):
    """Additional information on the MALA transition.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_probability
        The acceptance probability of the transition.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.

    """

    acceptance_probability: float
    is_accepted: bool


class SGLDState(NamedTuple):
    step: int
    position: PyTree
    logprob_grad: PyTree


