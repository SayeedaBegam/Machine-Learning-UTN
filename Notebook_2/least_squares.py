from typing import Callable
import numpy as np


def fit_line(data_x: np.ndarray, data_y: np.ndarray) -> tuple:
    """Use data_x and data_y to compute and return the optimal parameters m and
    b.
    Args:
        data_x (np.ndarray): the x coordinates of the dataset.
        data_y (np.ndarray): the y coordinates of the dataset.
    """
    assert len(data_x.shape) == 1
    assert data_x.shape == data_y.shape

    ####################################################################
    # YOUR CODE
    ####################################################################
    pass
    ####################################################################
    # END OF YOUR CODE
    ####################################################################


def line(x: np.ndarray, m: np.ndarray, b: np.ndarray):
    """Compute and return mx + b.
    Args:
        x (np.ndarray): an array of x values.
        m (float): slope.
        b (float): intercept.
    """

    ####################################################################
    # YOUR CODE
    ####################################################################

    pass

    ####################################################################
    # END OF YOUR CODE
    ####################################################################


def phi_poly(x: np.ndarray) -> np.ndarray:

    ####################################################################
    # YOUR CODE
    # TODO: Implement a fitting basis function.
    ####################################################################

    pass

    ####################################################################
    # END OF YOUR CODE
    ####################################################################


class LSRegression:

    def __init__(
        self,
        dim: int,
        m: int,
        phi: Callable[[np.ndarray], np.ndarray],
    ):
        """Initializes the least squares regression model.

        Args:
            dim (int): Dimensionality of the input data.
            m (int): Dimensionality of the feature space M.
            phi (Callable[[np.ndarray], np.ndarray]]):
                Phi function that maps the input data to the feature space.
        """
        self.dim = dim
        self.m = m
        self.weights = np.zeros(self.m)
        self.phi = phi

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """Fits the least squares regression model to the data.

        Args:
            X (np.ndarray): Input data of shape (N, D).
            y (np.ndarray): Target data of shape (N,).
        """
        # first dimension is N the number of samples
        assert X.shape[0] == y.shape[0]

        # second dimension is D the dimensionality of the samples
        assert X.shape[1] == self.dim

        # create design matrix
        Phi = self.Phi(X)

        # check dimensions of the design matrix
        assert Phi.shape[0] == X.shape[0]
        assert Phi.shape[1] == self.m

        ########################################################################
        # YOUR CODE
        # TODO: Implement the closed-form solution for linear regression.
        # Try to, avoid matrix inversion as it is numerically unstable.
        ########################################################################

        ########################################################################
        # END OF YOUR CODE
        ########################################################################

    def Phi(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Computes the design matrix Phi.

        Args:
            X (np.ndarray): Input data of shape (N, D).

        Returns:
            np.ndarray: Design matrix of shape (N, M).
        """

        ########################################################################
        # YOUR CODE
        # TODO: Implement the design matrix Phi
        ########################################################################

        pass

        ########################################################################
        # END OF YOUR CODE
        ########################################################################

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Computes the inference step of the linear regression model.

        Args:
            X (np.ndarray): Input data of shape (N, D).

        Returns:
            np.ndarray: Predictions of shape (N,).
        """

        ########################################################################
        # YOUR CODE
        # TODO: Implement the inference step of linear regression model
        ########################################################################

        pass

        ########################################################################
        # END OF YOUR CODE
        ########################################################################


class LSRidgeRegression(LSRegression):

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lmbda: float = 0.0,
    ):
        """Fits the ridge regression model to the data.

        Args:
            X (np.ndarray): Input data of shape (N, D).
            y (np.ndarray): Target data of shape (N,).
            lmbda (float, optional): Regularization parameter. Defaults to 0.0.
        """
        Phi = self.Phi(X)

        ########################################################################
        # YOUR CODE
        # TODO: Implement the closed-form solution for ridge regression using
        # the regularization parameter lmbda
        ########################################################################

        pass

        ########################################################################
        # END OF YOUR CODE
        ########################################################################


class LSClassification:

    def __init__(self, dim: int, k: int) -> None:
        """Initializes the least squares classification model.

        Args:
            dim (int): Dimensionality of the input data.
            k (int): Amount of classes K. Class labels are assumed to be in the
                range [0, K-1].
        """
        self.dim = dim
        self.k = k
        self.weights = np.zeros((self.dim + 1, self.k))

    def fit(self, X: np.ndarray, T: np.ndarray):
        """Fits the least squares classification model to the data.

        Args:
            X (np.ndarray): Input data of shape (N, D).
            T (np.ndarray): Target labels of shape (N, K).
        """
        # augment the data with a constant feature
        X_ = self.augment(X)

        ########################################################################
        # YOUR CODE
        # TODO: Implement the closed-form solution for least squares classification.
        ########################################################################

        pass

        ########################################################################
        # END OF YOUR CODE
        ########################################################################

    def augment(self, X: np.ndarray) -> np.ndarray:
        """Augments the data with a constant feature.

        Args:
            X (np.ndarray): Input data of shape (N, D).

        Returns:
            np.ndarray: Augmented data of shape (N, D+1).
        """

        ########################################################################
        # YOUR CODE
        # TODO: Add a column of ones to the input data.
        ########################################################################

        pass

        ########################################################################
        # END OF YOUR CODE
        ########################################################################

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute the inference step of the least squares classification model.

        Args:
            X (np.ndarray): Input data of shape (N, D).

        Returns:
            np.ndarray: Predictions of shape (N,).
        """

        ########################################################################
        # YOUR CODE
        # TODO: Predict the class labels for the input data. Remember to
        # augment the data first.
        ########################################################################

        pass

        ########################################################################
        # END OF YOUR CODE
        ########################################################################
