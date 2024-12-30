"""Gaussian process regression (GPR) applied to a real dataset."""
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def load_and_split(dataset: Path) -> tuple[np.ndarray]:
    """Load the dataset and split it into training and testing sets.

    Args:
        dataset (Path): the path to the dataset
    Returns:
        tuple: a tuple of numpy arrays (x_train, y_train, x_test, y_test)
    """

    ###########################################################################
    # YOUR CODE
    ###########################################################################
    pass
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################


def train_and_predict(dataset: Path) -> tuple[np.ndarray]:
    """Fit a GPR to the training split, predict for the testing split.

    Args:
        dataset (Path): the path to the dataset
    Returns:
        tuple: a tuple of numpy arrays (mean_prediction, std_prediction)
        containing the means and standard deviations of the prediction.
    """

    ########################################################################
    # YOUR CODE
    ########################################################################
    pass
    ########################################################################
    # END OF YOUR CODE
    ########################################################################
