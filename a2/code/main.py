#!/usr/bin/env python
import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import load_dataset, plot_classifier, handle, run, main
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_tree import RandomForest, RandomTree


@handle("1")
def q1():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    for k in [1, 3, 10]:
        model = KNN(k)
        model.fit(X, y)

        y_hat_train = model.predict(X)
        error_train = np.mean(y_hat_train != y)

        y_hat_test = model.predict(X_test)
        error_test = np.mean(y_hat_test != y_test)

        print(f"k={k}: training error = {error_train:.3f}, test error = {error_test:.3f}")

    model = KNN(1)
    model.fit(X,y)
    plot_classifier(model, X, y)

    fname = Path("..", "figs", "q1_knn_k1.pdf")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")

@handle("2")
def q2():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    ks = list(range(1, 30, 4))

    n = X.shape[0]
    n_folds = 10
    fold_size = n // n_folds

    cv_accs = np.zeros(len(ks))
    test_accs = np.zeros(len(ks))
    train_errs = np.zeros(len(ks))

    for index, k in enumerate(ks): #Enumerate makes it iterable
        fold_accs = np.zeros(n_folds)

        for fold in range(n_folds):
            mask = np.ones(n, dtype=bool)

            start = fold * fold_size
            end = start + fold_size

            mask[start:end] = False #False = Test

            X_train_fold = X[mask]
            y_train_fold = y[mask]
            X_validation = X[~mask]
            y_validation = y[~mask]

            model = KNN(k)
            model.fit(X_train_fold, y_train_fold)

            y_hat_fold = model.predict(X_validation)
            fold_accs[fold] = np.mean(y_hat_fold == y_validation)

        cv_accs[index] = np.mean(fold_accs)

        model = KNN(k)
        model.fit(X, y)

        y_hat_train = model.predict(X)
        train_errs[index] = np.mean(y_hat_train != y)

        y_hat_test = model.predict(X_test)
        test_accs[index] = np.mean(y_hat_test == y_test)

        print(f"k={k}: cv accuracy = {cv_accs[index]:.3f}, "
              f"test accuracy = {test_accs[index]:.3f}, "
              f"training error = {train_errs[index]:.3f}")

    best_cv_k = ks[np.argmax(cv_accs)]
    best_test_k = ks[np.argmax(test_accs)]
    print(f"Best k by CV accuracy: {best_cv_k}")
    print(f"Best k by test accuracy: {best_test_k}")

    #CV and Test Acc vs k
    plt.figure()
    plt.plot(ks, cv_accs, marker="o", label="Cross-Validation Accuracy")
    plt.plot(ks, test_accs, marker="o", label="Test Accuracy")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("CV and Test Accuracy vs. k")
    plt.legend()

    fname = Path("..", "figs", "q2_cv_test_accuracy.pdf")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")

    #Train Err vs k
    plt.figure()
    plt.plot(ks, train_errs, marker="o", color="tab:red", label="Training error")
    plt.xlabel("k")
    plt.ylabel("Training error")
    plt.title("Training error vs. k")
    plt.legend()

    fname = Path("..", "figs", "q2_train_error.pdf")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")






@handle("3.2")
def q3_2():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"].astype(bool)
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]
    groupnames = dataset["groupnames"]
    wordlist = dataset["wordlist"]

    """YOUR CODE HERE FOR Q3.2"""
    raise NotImplementedError()



@handle("3.3")
def q3_3():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    """CODE FOR Q3.4: Modify naive_bayes.py/NaiveBayesLaplace"""

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")


@handle("3.4")
def q3_4():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    """YOUR CODE HERE FOR Q3.4. Also modify naive_bayes.py/NaiveBayesLaplace"""
    raise NotImplementedError()



@handle("4")
def q4():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    print("Decision tree info gain")
    evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))

    """YOUR CODE FOR Q4. Also modify random_tree.py/RandomForest"""
    raise NotImplementedError()



@handle("5")
def q5():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("5.1")
def q5_1():
    X = load_dataset("clusterData.pkl")["X"]

    """YOUR CODE HERE FOR Q5.1. Also modify kmeans.py/Kmeans"""
    raise NotImplementedError()



@handle("5.2")
def q5_2():
    X = load_dataset("clusterData.pkl")["X"]

    """YOUR CODE HERE FOR Q5.2"""
    raise NotImplementedError()



if __name__ == "__main__":
    main()
