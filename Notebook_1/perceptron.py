import numpy as np

################################################################################
# YOUR CODE @T
# TODO: find a good value for eta @T
################################################################################
ETA = 0.0001  

################################################################################
# END OF YOUR CODE @T
################################################################################


class Perceptron:
    def __init__(self, dim: int = 2):
        """
        Args:
            dim (int, optional): Dimensionality of the perceptron.
                Defaults to 2.
        """
        self.dim: int = dim
        self.weights: np.ndarray = np.zeros(self.dim)
        self.bias: float = 0.0

    def activation(self, x: float) -> float:
        """Calculates the activation function of the perceptron."""
        ########################################################################
        # YOUR CODE @T
        # TODO: Implement the activation function of the perceptron @T
        ########################################################################

        pass  


        ########################################################################
        # END OF YOUR CODE @T
        ########################################################################

    def initialize_weights(self):
        """Initializes the weights of the perceptron."""
        ########################################################################
        # YOUR CODE @T
        # TODO: Initialize the weights uniform randomly in the interval @T
        # [-1, 1]. The bias should be initialized to 0 @T
        ########################################################################

        pass  


        ########################################################################
        # END OF YOUR CODE @T
        ########################################################################

    def predict_forloop(self, x: np.ndarray) -> float:
        """Predicts the label of a given sample using the weights and bias of
        the perceptron using Python for-loops.

        Args:
            x (np.ndarray): input sample of shape (dim,)

        Returns:
            float: Perceptron prediction after applying the activation function
        """
        ########################################################################
        # YOUR CODE @T
        # TODO: Implement the prediction of the perceptron using Python @T
        # for-loops @T
        ########################################################################

        pass  


        ########################################################################
        # END OF YOUR CODE @T
        ########################################################################

    def predict_vectorized(self, x: np.ndarray) -> float:
        """Predicts the label of a given sample using the weights and bias of
        the perceptron using vectorized operations from numpy.

        Args:
            x (np.ndarray): input sample of shape (dim,)

        Returns:
            float: Perceptron prediction after applying the activation function
        """
        ########################################################################
        # YOUR CODE @T
        # TODO: Implement the prediction of the perceptron using vectorized @T
        # operations from numpy @T
        ########################################################################

        pass  


        ########################################################################
        # END OF YOUR CODE @T
        ########################################################################

    def predict(self, x: np.ndarray, vectorized: bool = True) -> float:
        """Predicts the label of a given sample.

        Args:
            x (np.ndarray): input sample of shape (dim,)
            vectorized (bool, optional): Whether to use vectorized operations
            or Python for-loops. Defaults to True.

        Returns:
            float: Perceptron prediction for the given sample
        """
        if vectorized:
            return self.predict_vectorized(x)
        return self.predict_forloop(x)

    def update_step(
        self, x: np.ndarray, prediction: float, label: float, eta: float
    ):
        """Updates the weights and bias of the perceptron.

        Args:
            x (np.ndarray): _description_
            prediction (float): _description_
            label (float): _description_
            eta (float): _description_
        """
        ########################################################################
        # YOUR CODE @T
        # TODO: Implement the update rule as presented in the slides @T
        ########################################################################

        pass  


        ########################################################################
        # END OF YOUR CODE @T
        ########################################################################


def train_epoch(
    perceptron: Perceptron, X: np.ndarray, labels: np.ndarray, eta: float
) -> float:
    """Trains the perceptron (by updating its weights and bias) for one epoch
    (update steps for all samples contained in the dataset) and evaluates the
    perceptron on the data set by returning the average accuracy.

    Args:
        perceptron (Perceptron): Perceptron to be trained
        X (np.ndarray): Data set of shape (num_samples, dim)
        labels (np.ndarray): Labels of the data set of shape (num_samples,)
        eta (float): Learning rate

    Returns:
        float: Average accuracy of the perceptron on the data set
    """
    ############################################################################
    # YOUR CODE @T
    # TODO: Iterate over the whole data set (X) once and update the @T
    # perceptron. Return the accuracy of the perceptron at the end of the @T
    # epoch. @T
    ############################################################################

    pass  



    ############################################################################
    # END OF YOUR CODE @T
    ############################################################################


def fit(
    perceptron: Perceptron,
    X: np.ndarray,
    labels: np.ndarray,
    max_epochs: int,
    stop_accuracy: float,
    eta: float = ETA,
) -> list[float]:
    """Trains the perceptron for several epochs
    until a specific accuracy on the training data set
    has been reached or the maximum number of epochs has been reached.
    It returns the accuracies of the perceptron for each epoch in a Python list.

    Args:
        perceptron (Perceptron): Perceptron to be trained
        X (np.ndarray): Data set of shape (num_samples, dim)
        labels (np.ndarray): Labels of the data set of shape (num_samples,)
        max_epochs (int): Maximum number of epochs to train
        stop_accuracy (float): Stop training if the accuracy on the training set
            is above this threshold
        eta (float, optional): Learning rate. Defaults to ETA.

    Returns:
        list[float]: List of accuracies of the perceptron for each epoch
    """

    accuracies = []
    ############################################################################
    # YOUR CODE @T
    # TODO: Train the perceptron for at most max_epochs epochs until the @T
    # accuracy on the training data set (X) is above stop_accuracy. @T
    # Return an array which contains the accuracies for each epoch. @T
    ############################################################################

    pass  



    ############################################################################
    # END OF YOUR CODE @T
    ############################################################################
