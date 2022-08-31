# Author: Abdulrahman S. Omar<hsamireh@gmal.com>

from typing import Callable
from dmalax.types import PRNGKey, PyTree
import jax
import jax.numpy as jnp
from dmalax.state import SGLDState

__all__ = ["SGLDState", "init", "kernel"]


def init(position: PyTree, batch, grad_estimator_fn: Callable):
    logprob_grad = grad_estimator_fn(position, batch)
    return SGLDState(0, position, logprob_grad)


def kernel(grad_estimator_fn : Callable):
    """
    Build a DMALA kernel for binary variables - based on https://arxiv.org/abs/2206.09914 - Algorithm 2

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.
    """

    def diff_fn(pos, logprob_grad, step_size):

        theta = jax.tree_util.tree_map(lambda x, g: -0.5*(g)*(2.*x - 1) - (1./(2.*step_size)),
                                       pos, logprob_grad)

        return jax.nn.sigmoid(theta)

    def one_step(
            rng_key: PRNGKey, state: SGLDState, data_batch: PyTree, step_size: float) -> SGLDState:
        _, key_rmh = jax.random.split(rng_key)

        step, pos, logprob_grad = state
        u = jax.random.uniform(key_rmh, shape=state.position.shape)
        p_state = diff_fn(pos, logprob_grad, step_size)
        ind = jnp.array(u <= p_state)

        pos_new = (1. - pos)*ind + pos*(1. - ind)
        grad_state_new = grad_estimator_fn(pos_new, data_batch)

        return SGLDState(step+1, pos_new, grad_state_new)


    return one_step