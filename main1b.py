import numpy as np
import pandas as pd
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix


# TODO - Take care of ugly runtime warnings!
def main():
    # Load your dataset
    df = pd.read_csv("simple_classification.csv")
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    y = 2 * y - 1
    n_samples, n_features = X.shape

    # Parameters for the SVM
    C = 1.0  # Regularization parameter


    # Solve the dual problem and get alphas (assuming alphas is obtained as shown in the previous example)
    # Construct Q, q, G, h, A, b for the dual problem
    K = np.dot(X, X.T)  # Kernel matrix
    Q = csc_matrix(np.outer(y, y) * K)  # Q = y_i y_j x_i^T x_j
    q = -np.ones(n_samples)  # q = -1

    # Inequality constraints
    G = csc_matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
    h = np.hstack((np.zeros(n_samples), np.ones(n_samples) * C))

    # Equality constraints
    y = np.array(y)
    A = csc_matrix(y.reshape(1, -1))
    b = np.zeros(1)  # b = 0

    # Solve the quadratic program
    alphas = solve_qp(Q, q, G, h, A, b, solver='osqp')

    # Convert alphas to weights
    w = np.sum(alphas[:, None] * y[:, None] * X, axis=0)

    # Identify support vectors (alphas > threshold)
    threshold = 0.3
    support_vectors = alphas > threshold


    X, y = np.array(X), np.array(y)
    # Compute the bias using a support vector
    b = y[support_vectors][0] - np.dot(X[support_vectors][0], w)

    print("Weights:", w)
    print("Bias:", b)

    # Generate a range of x1 values
    x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

    # Compute corresponding x2 values for the decision boundary
    x2_boundary = - (w.iloc[0] / w.iloc[1]) * x1_range - b / w.iloc[1]

    # Compute the margin distance
    margin = 1 / np.sqrt(w.iloc[0] ** 2 + w.iloc[1] ** 2)

    # Compute corresponding x2 values for the margin lines
    x2_margin_pos = x2_boundary + margin
    x2_margin_neg = x2_boundary - margin

    # Plot the data points
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', label='Class -1')

    # Highlight support vectors
    plt.scatter(X[support_vectors, 0], X[support_vectors, 1], s=100, facecolors='none', edgecolors='k',
                label='Support Vectors')

    # Plot the decision boundary
    plt.plot(x1_range, x2_boundary, color='green', label='Decision Boundary')

    # Plot the margin lines
    plt.plot(x1_range, x2_margin_pos, color='grey', linestyle='--', label='Margin')
    plt.plot(x1_range, x2_margin_neg, color='grey', linestyle='--')

    # Labeling the plot
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('SVM Decision Boundary with Margins and Support Vectors')

    # Show the plot
    plt.show()
    """# Load your dataset
    df = pd.read_csv("simple_classification.csv")
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    y = 2*y - 1
    n_samples, n_features = X.shape

    # Parameters for the SVM
    C = 1.0  # Regularization parameter

    # Construct Q, q, G, h, A, b for the dual problem
    K = np.dot(X, X.T)  # Kernel matrix
    Q = np.outer(y, y) * K  # Q = y_i y_j x_i^T x_j
    q = -np.ones(n_samples)  # q = -1

    # Inequality constraints
    G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
    h = np.hstack((np.zeros(n_samples), np.ones(n_samples) * C))

    # Equality constraints
    y = np.array(y)
    A = y.reshape(1, -1)  # A = y_i
    b = np.zeros(1)  # b = 0

    # Solve the quadratic program
    alphas = solve_qp(Q, q, G, h, A, b, solver='osqp')

    print("Lagrange multipliers (alphas):", alphas)"""


if __name__ == "__main__":
    main()