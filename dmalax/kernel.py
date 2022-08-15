
from typing import Callable
from .types import PyTree, PRNGKey
from .base import SamplingAlgorithm
from .dmala import init, kernel

class dmala:
    """Implements the (basic) user interface for the DMALA kernel.

    Examples
    --------

    A new DMALA kernel can be initialized and used with the following code:

    .. code::

        dmala = dmalax.dmala(logprob_fn, step_size)
        state = dmala.init(position)
        new_state, info = dmala.step(rng_key, state)

    Kernels are not jit-compiled by default, so you will need to do it manually:

    .. code::

       step = jax.jit(dmala.step)
       new_state, info = step(rng_key, state)

    Should you need to you can always use the base kernel directly:

    .. code::

       kernel = dmalax.dmala.kernel(logprob_fn)
       state = dmalax.dmala.init(position, logprob_fn)
       state, info = kernel(rng_key, state, logprob_fn, step_size)

    Parameters
    ----------
    logprob_fn
        The logprobability density function we wish to draw samples from. This
        is minus the potential function.
    step_size
        The value to use for the step size in the symplectic integrator.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """

    init = staticmethod(init)
    kernel = staticmethod(kernel)

    def __new__(cls,
            logprob_fn: Callable,
            step_size: float,
    ) -> SamplingAlgorithm:
        step = cls.kernel()

        def init_fn(position: PyTree):
            return cls.init(position, logprob_fn)

        def step_fn(rng_key: PRNGKey, state):
            return step(rng_key, state, logprob_fn, step_size)

        return SamplingAlgorithm(init_fn, step_fn)
