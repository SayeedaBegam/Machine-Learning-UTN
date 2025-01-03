from torch import nn


class SGDOptimizer:
    def __init__(self, model: nn.Module, lr: float = 0.01):
        """
        Args:
            model (nn.Module): Network model
            lr (float, optional): Learning rate. Defaults to 0.01.
        """

        ####################################################################
        # YOUR CODE
        ####################################################################
        pass
        ####################################################################
        # END OF YOUR CODE
        ####################################################################

    def step(self):
        """Updates the parameters of the model using the models gradients."""

        ####################################################################
        # YOUR CODE
        # Hint: Figure out how to access the parameters a nn.Module object.
        ####################################################################
        pass
        ####################################################################
        # END OF YOUR CODE
        ####################################################################

    def zero_grad(self):
        """Sets the gradients of all model parameters to None."""

        ####################################################################
        # YOUR CODE
        ####################################################################
        pass
        ####################################################################
        # END OF YOUR CODE
        ####################################################################
