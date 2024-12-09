import numpy as np
import pandas as pd
from SVM import SVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(filepath, scatter_plt=False, transform=False):
    df = pd.read_csv(filepath)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X))
    print("Labels before normalization -\n", y)
    if transform:
        y = 2 * y - 1
    if scatter_plt:
        model = SVM(kernel="linear")
        model.plot_data(X, y)
    return train_test_split(X, y, test_size=0.2, random_state=41)


def main():
    X_train, X_test, y_train, y_test = preprocess("simple_nonlin_classification.csv")
    model = SVM(kernel='poly', degree=7)
    model.fit(X_train, y_train, thresh=0.01)

    # model.plot_error(X_train, y_train, X_test, y_test, 'degree', range(4, 8), thresh=1e-6)
    # model.plot_decision_boundary(X_train, y_train)
    predictions = model.predict(X_test)
    print("y test:\n", y_test, "Predictions:\n", predictions)
    print("polynomial kernels score: ", model.score(X_test, y_test))



if __name__ == "__main__":
    main()
