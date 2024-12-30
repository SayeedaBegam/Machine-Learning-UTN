import torch
from torch import nn


class Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logit: torch.Tensor):
        # for numeric stability we use the tanh formula here
        out = 1 / 2 * (1 + torch.tanh(logit / 2))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (out,) = ctx.saved_tensors
        return grad_output * out * (1 - out)


class ReLU(torch.autograd.Function):

    ####################################################################
    # YOUR CODE
    ####################################################################
    pass
    ####################################################################
    # END OF YOUR CODE
    ####################################################################


class BinaryCrossEntropy(torch.autograd.Function):
    # mean reduction

    ####################################################################
    # YOUR CODE
    # TODO: Implement forward and backward functions. Keep in mind that you are receiving batches.
    # The input shape will be (N, 1). The reduction method should be `mean`. Use d_labels=None
    # for the derivative with respect to the labels.
    ####################################################################
    pass
    ####################################################################
    # END OF YOUR CODE
    ####################################################################


class CrossEntropy(torch.autograd.Function):

    ####################################################################
    # YOUR CODE
    # TODO: This class combines softmax and and the Categorical Cross Entropy loss.
    # Implement forward and backward functions. Keep in mind that you are receiving batches.
    # The input shape will be (N, C). The reduction method should be `mean`. Use d_labels=None
    # for the derivative with respect to the labels. Use the log-sum-exp trick to avoid
    # numerical instability. Think how you can reformulate the joined formula to
    # get a log-sum-exp expression.  Why can the softmax be problematic in terms of
    # numerical stability? Answer with a comment below.
    ####################################################################
    pass
    ####################################################################
    # END OF YOUR CODE
    ####################################################################


# Linear/fully connected layer
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ####################################################################
        # YOUR CODE
        # TODO: Create parameters using nn.Parameter
        ####################################################################

        ####################################################################
        # END OF YOUR CODE
        ####################################################################

        self.reset_parameters()

    def reset_parameters(self) -> None:

        ####################################################################
        # YOUR CODE
        # TODO: Initialize parameters as follows w with the standard distribution
        # and b constant to zero
        ####################################################################
        pass
        ####################################################################
        # END OF YOUR CODE
        ####################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_features)

        Returns:
            torch.Tensor: Computed output tensor of shape (N, out_features)
        """

        ####################################################################
        # YOUR CODE
        # TODO: Implement the forward pass of the linear layer. Be careful
        # to handle batches correctly. Use broadcasting to avoid for-loops.
        ####################################################################
        pass
        ####################################################################
        # END OF YOUR CODE
        ####################################################################


# Wrap layers into nn.module (nothing to do here)


class ReLULayer(nn.Module):
    def forward(self, x):
        return ReLU.apply(x)


class BinaryCrossEntropyLoss(nn.Module):
    """Combines sigmoid function and binary cross entropy loss"""

    def forward(self, logits, labels):
        return BinaryCrossEntropy.apply(Sigmoid.apply(logits), labels)


class CategoricalCrossEntropyLoss(nn.Module):
    """Combines softmax function and categorical cross entropy loss"""

    def forward(self, logits, labels):
        return CrossEntropy.apply(logits, labels)
