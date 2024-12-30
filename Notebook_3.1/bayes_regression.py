from typing import Callable
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats._multivariate import multivariate_normal_frozen


def likelihood(
    X: np.ndarray,
    t: np.ndarray,
    w: np.ndarray,
    beta: float,
    phi: Callable[[np.ndarray], np.ndarray],
) -> float:
    """Compute the likelihood of seeing the targets t given the weights w.

    Args:
        X (np.ndarray): Independent variables of shape (N, D)
        t (np.array): Targets (dependent variables) of shape (N, 1)
        w (np.ndarray): Vector whose length is determined by the output of phi
        beta (float): noise precision scalar
        phi (Callable[np.ndarray, [np.ndarray]]): a function to augment the
        inputs x with features.

    Hint: Use the scipy function multivariate_normal to compute the pdf of the
    multivariate gaussian.

    Returns:
        float: the computed likelihood.
    """

    ########################################################################
    # YOUR CODE
    ########################################################################

    pass

    ########################################################################
    # END OF YOUR CODE
    ########################################################################


def posterior_distribution(
    X: np.ndarray,
    t: np.ndarray,
    alpha: float,
    beta: float,
    phi: Callable[[np.ndarray], np.ndarray],
) -> multivariate_normal_frozen:
    """Compute the posterior distribution.

    Args:
       X (np.ndarray): Independent variables of shape (N, D)
       t (np.array): Targets (dependent variables) of shape (N, 1)
       w (np.ndarray): Vector whose length is determined by the output of
           phi
       beta (float): noise precision scalar
       phi (Callable[np.ndarray, [np.ndarray]]): a function to augment the
           inputs x with features.
    Returns: multivariate_normal_frozen: an instance of the
    multivariate_normal class where mean and cov have already been set

    Hint: you can use np.linalg.inv to invert the matrix as it is done in
    the formula. In practice you should never invert large matrices that are
    not diagonal due to numerical unstability.
    """

    ########################################################################
    # YOUR CODE
    ########################################################################

    pass

    ########################################################################
    # END OF YOUR CODE
    ########################################################################


def predictive_distribution(
    x: np.ndarray,
    X: np.ndarray,
    t: np.ndarray,
    alpha: float,
    beta: float,
    phi: Callable[[np.ndarray], np.ndarray],
) -> multivariate_normal_frozen:
    """Compute the predictive distribution for x.

    Args:
        x (np.ndarray): A single element x. Has the same shape as the
            elements in X.
        X (np.ndarray): Independent variables of shape (N, D)
        t (np.array): Targets (dependent variables) of shape (N, 1)
        alpha (float): precision scalar for the weights
        beta (float): noise precision scalar
        phi (Callable[np.ndarray, [np.ndarray]]): a function to augment the
            inputs x with features.
    """

    ########################################################################
    # YOUR CODE
    ########################################################################

    pass

    ########################################################################
    # END OF YOUR CODE
    ########################################################################
