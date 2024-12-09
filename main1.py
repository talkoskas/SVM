import numpy as np
import pandas as pd
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix


# TODO - Delete unnecessary prints and add necessary prints
def main():
    df = pd.read_csv("simple_classification.csv")
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    y = 2*y - 1
    N, samples = X.shape[0], X.shape[1]

    # Construct P (identity matrix with an extra row and column for the bias term)
    P = np.eye(samples + 1)
    P[-1, -1] = 0
    P = csc_matrix(P)

    # Construct q (zero vector - no linear constraints in the primal problem)
    q = np.zeros(samples + 1)
    # Construct G and h
    G = pd.DataFrame(np.diag(y)@X)
    G.insert(loc=2, column='yi', value=y)
    G = -1*G
    G = csc_matrix(np.array(G))
    h = -np.ones(y.shape[0])

    x = solve_qp(P, q, G, h, solver='osqp')
    print("Solution - \n", x)
    w1, w2, b = x
    Xx = X.values

    # Generate a range of x1 values
    x1_range = np.linspace(Xx[:, 0].min(), Xx[:, 0].max(), 100)

    # Compute corresponding x2 values for the decision boundary
    x2_boundary = - (w1 / w2) * x1_range - b / w2

    # Compute the margin distance
    margin = 1 / np.sqrt(w1 ** 2 + w2 ** 2)

    # Compute corresponding x2 values for the margin lines
    x2_margin_pos = x2_boundary + margin
    x2_margin_neg = x2_boundary - margin

    # Plot the decision boundary
    plt.plot(x1_range, x2_boundary, color='green', label='Decision Boundary')

    # Plot the margin lines
    plt.plot(x1_range, x2_margin_pos, color='grey', linestyle='--', label='Margin')
    plt.plot(x1_range, x2_margin_neg, color='grey', linestyle='--')

    # Labeling the plot
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('SVM Decision Boundary')
    # Scatter plot the data points with different colors based on their labels
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.bwr, marker='o', edgecolors='k')
    # Set labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter Plot of Data Points with Weight Vector')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()