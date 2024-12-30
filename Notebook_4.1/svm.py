from cvxopt import matrix, solvers
import numpy as np
import matplotlib.pyplot as plt

# Threshold for choosing support vectors.
ALPHA_TOL = 1e-5


def solve_dual_svm(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve the dual formulation of the SVM problem.

    Use the already imported functions from cvxopt to solve the QP problem
    of the dual SVM problem.

    Hint: cvxopt needs the matrices as objects of type `cvxopt.matrix`
    which accepts numpy arrays as input.

    Args:
        X (np.ndarray): Input features. Shape: (N, D)
        y (np.ndarray): Binary class labels (in {-1, 1} format). Shape: (N,)

    Returns:
        np.ndarray: Solution of the dual problem. Shape: (N,)
    """

    ####################################################################
    # YOUR CODE
    ####################################################################
    pass
    ####################################################################
    # END OF YOUR CODE
    ####################################################################



def compute_weights_and_bias(
    alpha: np.ndarray, X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, float]:
    """Recover the weights w and the bias b using the dual solution alpha.

    Hint: Use the global variable ALPHA_TOL as a threshold for choosing
    the support vectors.

    Args:
        alpha (np.ndarray): Solution of the dual problem. Shape: (N,)
        X (np.ndarray): Input features. Shape: (N, D)
        y (np.ndarray): Binary class labels (in {-1, 1} format). Shape: (N,)

    Returns:
        tuple[np.ndarray, float]: Weight vector w. Shape: (D,); Bias term b.
    """

    ####################################################################
    # YOUR CODE
    ####################################################################
    pass
    ####################################################################
    # END OF YOUR CODE
    ####################################################################


def plot_data_with_hyperplane_and_support_vectors(
    X: np.ndarray, y: np.ndarray, alpha: np.ndarray, w: np.ndarray, b: float
):
    """Plot the data as a scatter plot together with the separating hyperplane.

    Args:
        X (np.ndarray): Input features. Shape: (N, D)
        y (np.ndarray): Binary class labels (in {-1, 1} format). Shape: (N,)
        alpha (np.ndarray): Solution of the dual problem. Shape: (N,)
        w (np.ndarray): Weight vector. Shape: (D,)
        b (float): Bias term.
    """
    plt.figure(figsize=[10, 8])
    # Plot the hyperplane
    slope = -w[0] / w[1]
    intercept = -b / w[1]
    x = np.linspace(X[:, 0].min(), X[:, 0].max())
    plt.plot(x, x * slope + intercept, "k-", label="decision boundary")
    plt.plot(x, x * slope + intercept - 1 / w[1], "k--")
    plt.plot(x, x * slope + intercept + 1 / w[1], "k--")
    # Plot all the datapoints
    plt.scatter(X[:, 0], X[:, 1], c=y)
    # Mark the support vectors
    support_vecs = alpha > ALPHA_TOL
    plt.scatter(
        X[support_vecs, 0],
        X[support_vecs, 1],
        c=y[support_vecs],
        s=250,
        marker="*",
        label="support vectors",
    )
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend(loc="upper left")
