
from typing import Callable
from dmalax.types import PyTree, PRNGKey
from dmalax.base import SamplingAlgorithm
import dmalax.mcmc as mcmc


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

    init = staticmethod(mcmc.dmala.init)
    kernel = staticmethod(mcmc.dmala.kernel)

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

class sgdld:
    """Implements the (basic) user interface for the SGDLD kernel.

    The general sgdld kernel (:meth:`dmalax.mcmc.sgdld.kernel`, alias `dmalax.sgdld.kernel`) can be
    cumbersome to manipulate. Since most users only need to specify the kernel
    parameters at initialization time, we provide a helper function that
    specializes the general kernel.

    Example
    -------

    To initialize a SGDLD kernel one needs to specify a schedule function, which
    returns a step size at each sampling step, and a gradient estimator
    function. Here for a constant step size, and `data_size` data samples:

    .. code::

        schedule_fn = lambda _: 1e-3
        grad_fn = blackjax.sgmcmc.gradients.grad_estimator(logprior_fn, loglikelihood_fn, data_size)

    We can now initialize the sgdld kernel and the state, assuming we have an iterator `batches` that yields batches of data we can perform one step:

    .. code::
        sgdld = dmalax.sgdld(grad_fn, schedule_fn)
        state = dmalax.init(position, data_batch)
        data_batch = next(batches)
        new_state = sgdld.step(rng_key, state, data_batch)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

       step = jax.jit(sgdld.step)
       new_state, info = step(rng_key, state)

    Parameters
    ----------
    gradient_estimator_fn
       A function which, given a position and a batch of data, returns an estimation
       of the value of the gradient of the log-posterior distribution at this position.
    schedule_fn
       A function which returns a step size given a step number.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """

    init = staticmethod(mcmc.sgdld.init)
    kernel = staticmethod(mcmc.sgdld.kernel)

    def __new__(  # type: ignore[misc]
            cls,
            grad_estimator_fn: Callable,
            schedule_fn: Callable,
    ) -> SamplingAlgorithm:

        step = cls.kernel(grad_estimator_fn)

        def init_fn(position: PyTree, data_batch: PyTree):
            return cls.init(position, data_batch, grad_estimator_fn)

        def step_fn(rng_key: PRNGKey, state, data_batch: PyTree):
            step_size = schedule_fn(state.step)
            return step(rng_key, state, data_batch, step_size)

        return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]