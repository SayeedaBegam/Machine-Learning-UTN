import numpy as np
from sklearn.linear_model import LinearRegression


def ls_regression_scikit(
        X: np.ndarray, y: np.ndarray, predict_until: int
) -> np.ndarray:
    """Fit a least squares regression model using scikit-learn.

    Args:
        X (np.ndarray): The input data of shape (n_samples, n_features).
        y (np.ndarray): The labels of shape (n_samples,).
        predict_until (int): the date until which you should predict the
        extreme temperatures.

    Returns:
        np.ndarray: The predictions of the fitted model until 'predict_until'
    """

    ############################################################################
    # YOUR CODE
    # TODO: Fit a least squares regression model using scikit-learn.
    # Then use the fitted model to predict the evolution of extreme weather
    # from 2000 until the year given by the 'predict_until' parameter.
    ############################################################################

    pass

    ############################################################################
    # END OF YOUR CODE
    ############################################################################
