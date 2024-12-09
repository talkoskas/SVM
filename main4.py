import numpy as np
#from sklearn.svm import SVC as SVM
from SVM import SVM
import pandas as pd
from sklearn.model_selection import train_test_split


def poly_kernel(x, y, i: int = 2):
    return (1 + np.dot(x.T, y)) ** i


def rbf_kernel(x, y, gamma=1.0):
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)


def sigmoid_kernel(x, y, alpha=0.1, c=0.0):
    return np.tanh(alpha * np.dot(x, y) + c)

def preprocess(filepath, scatter_plt=False):
    df = pd.read_csv(filepath)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    y = 2 * y - 1
    if scatter_plt:
        model = SVM(kernel="linear")
        model.plot_data(X, y)
    return train_test_split(X, y, test_size=0.2, random_state=41)

def main():
    X_train, X_test, y_train, y_test = preprocess("Processed Wisconsin Diagnostic Breast Cancer.csv")
    print("Types - X: train - ", type(X_train), "\ntest - ", type(X_test))
    print("y: train - ", type(y_train), "\ntest - ", type(y_test))
    print("Shapes:\nX_train - ", X_train.shape, "\ny_train - ", len(y_train))
    print("X_test - ", X_test.shape, "\ny_test - ", len(y_test))

    model = SVM(kernel="rbf", gamma=0.001)
    model.fit(X_train, y_train, thresh=1e-6)
    # The below command of plot_error is in note because a screenshot was added,
    # if you wish to test it, take it out of note AND put the above fit in note to avoid
    # unnecessary usage of fit method!
    # model.plot_error(X_train, y_train, X_test, y_test, 'gamma', np.logspace(-3, 2, 6), thresh=1e-6)
    predictions = model.predict(X_test)
    print("y test:\n", y_test, "Predictions:\n", predictions)
    score = model.score(X_test, y_test)
    print("SVM model finished with score of:", score)


if __name__ == "__main__":
    main()
