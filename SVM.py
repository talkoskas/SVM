import numpy as np
import pandas as pd
from qpsolvers import solve_qp
from scipy.sparse import csc_matrix
import itertools
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, kernel, degree=3, C=1.0, gamma='scale'):
        if isinstance(kernel, str):
            if kernel == "poly":
                self.kernel_ = self.poly_ker
            elif kernel.lower() == "rbf":
                self.kernel_ = self.rbf_ker
            elif kernel.lower() == "sigmoid":
                self.kernel_ = self.sigmoid_ker
            elif kernel.lower() == "linear":
                self.kernel_ = self.linear_ker
        else:
            self.kernel_ = kernel
        self.degree_ = degree
        self.C_ = C
        self.gamma_ = gamma
        self.support_vectors_ = None
        self.alpha_ = None
        self.intercept_ = None
        self.X_ = None
        self.y_ = None

    def poly_ker(self, x, y):
        return (1 + np.dot(x, y)) ** self.degree_

    def rbf_ker(self, x, y):
        gamma = np.mean(np.float64(self.gamma_))
        return np.exp(-gamma * np.linalg.norm(x - y) ** 2)

    def sigmoid_ker(self, x, y):
        alpha = float(self.gamma_)
        c = 1.0
        return np.tanh(alpha * np.dot(x, y) + c)

    def linear_ker(self, x, y):
        return np.dot(x, y)

    def svm_dual_kernel(self, X, y, ker, max_iter=4000, verbose=False):
        N = X.shape[0]
        X = X.astype(np.float64)

        P = np.empty((N, N))
        for i, j in itertools.product(range(N), range(N)):
            P[i, j] = y[i] * y[j] * ker(X.iloc[i, :], X.iloc[j, :])

        P = 0.5 * (P + P.T)
        P = 0.5 * P
        P = csc_matrix(P)

        q = -np.ones(N)
        GG = np.vstack([-np.eye(N), np.eye(N)])
        h = np.hstack([np.zeros(N), np.ones(N) * self.C_])

        alpha = solve_qp(P, q, GG, h, solver='osqp', max_iter=max_iter, verbose=verbose)

        return alpha

    def support_vectors(self, alpha, thresh=0.0001):
        return np.argwhere(np.abs(alpha) > thresh).reshape(-1)

    def fit(self, X, y, max_iter=4000, thresh=0.0001, verbose=False):
        self.X_ = X.astype(np.float64)
        self.y_ = np.array(y).astype(np.float64)

        if self.gamma_ == 'scale':
            self.gamma_ = 1 / np.mean(X.shape[0] * np.var(self.X_, axis=0))
        elif self.gamma_ == 'auto':
            self.gamma_ = 1 / X.shape[0]

        self.alpha_ = self.svm_dual_kernel(self.X_, self.y_, self.kernel_, max_iter=max_iter, verbose=verbose)

        print("Alpha values after optimization:", self.alpha_)

        support_vectors = self.support_vectors(self.alpha_, thresh)
        print("Support vectors:\n", support_vectors)

        if len(support_vectors) > 0:
            idx = support_vectors[0]
            kernel_values = np.array([self.kernel_(self.X_.iloc[i, :].values, self.X_.iloc[idx, :].values) for i in range(self.X_.shape[0])])
            self.intercept_ = self.y_[idx] - np.sum(self.alpha_ * self.y_ * kernel_values)
        else:
            self.intercept_ = 0

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float64)
        else:
            X = np.array(X).astype(np.float64)

        if isinstance(self.X_, pd.DataFrame):
            self.X_ = self.X_.values.astype(np.float64)

        K = np.array([[self.kernel_(self.X_[i], x) for i in range(self.X_.shape[0])] for x in X])
        decision = np.dot(K, (self.alpha_ * self.y_)) + self.intercept_
        return np.sign(decision)

    def decision_function(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float64)
        else:
            X = np.array(X).astype(np.float64)

        if isinstance(self.X_, pd.DataFrame):
            self.X_ = self.X_.values.astype(np.float64)

        K = np.array([[self.kernel_(self.X_[i], x) for i in range(self.X_.shape[0])] for x in X])
        return np.dot(self.alpha_ * self.y_, K.T) + self.intercept_

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def plot_data(self, X, y):
        """
        Scatter plot of data points.

        Parameters:
        - X: Features matrix.
        - y: Target vector.
        """
        X = np.array(X)
        plt.figure(figsize=(8, 6))
        plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Scatter Plot of Data Points')
        plt.legend()
        plt.show()

    def plot_error(self, X_train, y_train, X_val, y_val, param_name, param_range, thresh=1e-5):
        errors = []

        for param_value in param_range:
            setattr(self, f'{param_name}_', param_value)  # Update the hyperparameter
            self.fit(X_train, y_train, thresh=thresh)
            score = self.score(X_val, y_val)
            errors.append(1 - score)

        plt.figure(figsize=(10, 6))
        plt.plot(param_range, errors, marker='o')
        plt.title(f'Error vs {param_name}')
        plt.xlabel(param_name)
        plt.ylabel('Error')
        plt.grid(True)
        plt.show()

    def plot_decision_boundary(self, X, y):
        # Create a mesh to plot the decision boundary
        h = .02  # step size in the mesh
        X = pd.DataFrame(X)
        x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = self.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

        support_vector_indices = self.support_vectors(self.alpha_)
        plt.scatter(X.iloc[support_vector_indices, 0], X.iloc[support_vector_indices, 1], s=100,
                    facecolors='none', edgecolors='k', label='Support Vectors')

        # Compute the decision function for the mesh grid
        decision_function = self.decision_function(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()]))
        decision_function = decision_function.reshape(xx.shape)

        # Plot the decision boundary and margins
        plt.contour(xx, yy, decision_function, levels=[0], linewidths=2, colors='k')
        plt.contour(xx, yy, decision_function, levels=[-1, 1], linewidths=2, colors='gray', linestyles='dashed')

        plt.title(f'SVM Decision Boundary with {self.kernel_.__name__} Kernel')
        plt.legend()
        plt.show()

# Example usage:
# model = SVM(kernel="poly", degree=3, C=1.0)
# model.fit(X_train, y_train)
# score = model.score(X_test, y_test)
