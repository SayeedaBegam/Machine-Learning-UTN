from typing import Optional
import numpy as np
from collections import defaultdict
from sklearn.naive_bayes import BernoulliNB


def bernoulli_naive_bayes_scikit(
    X_train: np.ndarray,
    Y_train: list[str],
    X_test: np.ndarray,
    smoothing_factor: float = 1.0,
) -> tuple[list[str], list[dict[str, float]]]:
    """Train a Bernoulli Naive Bayes classifier using scikit-learn.

    Hint: Use the already implemented BernoulliNB class from scikit-learn.

    Args:
        X_train (np.ndarray): Train data of shape (n_samples, n_features).
        Y_train (list[str]): Labels of shape (n_samples,).
        X_test (np.ndarray): Test data of shape (n_samples, n_features).
        smoothing_factor (float, optional): Laplace smoothing factor for the likelihood.
            Defaults to 1.0.

    Returns:
        tuple[list[str], list[dict[str, float]]]: A tuple containing the
            predicted classes and the posterior accuracies,
            e.g. (['Y'], [{'Y': 0.92, 'N': 0.07}])
    """

    ########################################################################
    # YOUR CODE
    ########################################################################

    pass

    ########################################################################
    # END OF YOUR CODE
    ########################################################################


class BernoulliNaiveBayes:
    def __init__(self) -> None:
        self.priors: Optional[dict[str, float]] = None
        self.likelihood: Optional[dict[str, np.ndarray]] = None

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: list[str],
        uniform_priors: bool = False,
        smoothing_factor: float = 1.0,
    ):
        """Fit the model to the data.

        Hint: Use self.priors and self.likelihood to store the priors and the likelihood
        for later use in the predict method.

        Args:
            X_train (np.ndarray): Training data of shape (n_samples,
                n_features).
            Y_train (list[str]): Labels of shape (n_samples,).
            uniform_priors (bool, optional): Whether to use uniform priors.
                Defaults to False.
            smoothing_factor (float, optional): Laplace smoothing factor for
                the likelihood. Defaults to 1.0.
        """

        ########################################################################
        # YOUR CODE
        ########################################################################

        pass

        ########################################################################
        # END OF YOUR CODE
        ########################################################################

    def predict(
        self, X_test: np.ndarray
    ) -> tuple[list[str], list[dict[str, float]]]:
        """Predict the class of the given data.


        Args:
            X_test (np.ndarray): Test data of shape (n_samples, n_features).


        Returns:
            tuple[list[str], list[dict[str, float]]]: A tuple containing the
                predicted classes and the posterior accuracies,
                e.g. (['Y'], [{'Y': 0.92, 'N': 0.07}])
        """
        if self.priors is None or self.likelihood is None:
            raise RuntimeError("You need to fit the model first.")

        ########################################################################
        # YOUR CODE
        ########################################################################

        pass

        ########################################################################
        # END OF YOUR CODE
        ########################################################################

    def _label_indices(self, labels: list[str]) -> dict[str, list[int]]:
        """Get the indices of the labels.

        Args:
            labels (list[str]): The labels, e.g. ["Y", "N", "Y", "Y"].

        Returns:
            dict[str, int]: A dictionary mapping each label to its index,
                e.g. {"Y": [0, 2, 3], "N": [1]}.
        """

        ########################################################################
        # YOUR CODE
        ########################################################################

        pass

        ########################################################################
        # END OF YOUR CODE
        ########################################################################

    def _priors(
        self, label_indices: dict[str, list[int]], uniform: bool = False
    ) -> dict[str, float]:
        """Compute the priors for each class.


        Args:
            label_indices (dict[str, list[int]]): A dictionary mapping each
                label to its indices. Output from _label_indices method.
            uniform (bool, optional): Whether to use uniform priors. Defaults
                to False.

        Returns:
            dict[str, float]: A dictionary mapping each label to its prior.
        """

        ########################################################################
        # YOUR CODE
        ########################################################################

        pass

        ########################################################################
        # END OF YOUR CODE
        ########################################################################

    def _likelihood(
        self,
        features: np.ndarray,
        label_indices: dict[str, list[int]],
        smoothing=0,
    ) -> dict[str, np.ndarray]:
        """Compute the likelihood for each class using the given features.

        Hint: Don't forget to apply Laplace smoothing.

        Args:
            features (np.ndarray): The features of shape (n_samples,
                n_features).
            label_indices (dict[str, list[int]]): A dictionary mapping each
                label to its indices.
            smoothing (int, optional): Laplace smoothing factor. Defaults to 0.

        Returns:
            dict[str, np.ndarray]: A dictionary mapping each label to its
                likelihood (multi-dimensional).
        """

        ########################################################################
        # YOUR CODE
        ########################################################################

        pass

        ########################################################################
        # END OF YOUR CODE
        ########################################################################

    def _posteriors(
        self,
        X: np.ndarray,
        priors: dict[str, float],
        likelihood: dict[str, np.ndarray],
    ) -> list[dict[str, float]]:
        """Compute the posterior for each class using the given features.

        Args:
            X (np.ndarray): The features of shape (n_samples, n_features).
            priors (dict[str, float]): The priors for each class.
            likelihood (dict[str, np.ndarray]): The likelihood for each class.

        Returns:
            list[dict[str, float]]: A list of dictionaries mapping each label to
                its posterior. Each dictionary should sum up to 1. The list
                should have the same length as the number of samples.
        """

        ########################################################################
        # YOUR CODE
        ########################################################################

        pass

        ########################################################################
        # END OF YOUR CODE
        ########################################################################
