"""Contains helper functions for autograd."""

from collections.abc import Callable
from functools import wraps


def restore_dims_from_vmap(func: Callable, batch_over_dims: tuple[int, ...]) -> Callable:
    """
    Helper function to restore the dimensions of a tensor input
    to a function that were removed by torch.vmap. See Notes.

    Parameters
    ----------

    func : Callable
        The function to wrap for restored dimensions.
    batch_over_dims : tuple[int, ...]
        The dimensions that are vmaped over. See Notes

    Returns
    -------

    Callable
        The wrapped function with restored dimensions.

    Notes
    -----

    When, e.g., wanting batch dimensions for PyTorch autograd functions,
    one needs to vmap, but still wants to call the function with the same
    tensor dimensions.
    E.g., when computing the jacobian of a vector function that takes a tensor
    of shape (n_batch, n_dim), one needs to vmap over n_batch, but
    the function should still be called with a tensor of shape
    (n_batch, n_dim) even if n_batch=1. Vmap would remove the batch dimension.
    This function restores the dimensions that get removed by vmap,
    for the function.

    Only supports vmaping over the same dimensions for all inputs.
    E.g. a call of vmap twice (both times with in_dims=0) corresponds to
    batch_over_dims=(0, 0).
    """

    @wraps(func)
    def function_with_restored_dims(*x):
        x = list(x)
        for i in range(len(x)):
            for j in batch_over_dims:
                x[i] = x[i].unsqueeze(j)
        computed_result = func(*x)

        if not isinstance(computed_result, tuple):
            for j in batch_over_dims:
                computed_result = computed_result.squeeze(j)
            return computed_result

        computed_result = list(computed_result)
        for i in range(len(computed_result)):
            for j in batch_over_dims:
                computed_result[i] = computed_result[i].squeeze(j)
        return tuple(computed_result)

    return function_with_restored_dims
