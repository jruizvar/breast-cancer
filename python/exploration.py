"""
    Exploration of Breast Cancer Data Set

    https://archive.ics.uci.edu/ml/datasets/breast+cancer
"""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer, StandardScaler

import os
import pandas as pd


def get_data(path):
    df = pd.read_csv(path)
    X = df.loc[:, df.columns != 'recurrence']
    X = pd.get_dummies(X)
    lb = LabelBinarizer()
    y = lb.fit_transform(df.recurrence).ravel()
    return X, y


def get_scores(clf, X, y):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_validate(clf, X, y,
                            scoring=['accuracy', 'roc_auc'],
                            cv=cv,
                            return_train_score=True)
    train_accuracy = scores['train_accuracy'].mean()
    test_accuracy = scores['test_accuracy'].mean()
    train_roc_auc = scores['train_roc_auc'].mean()
    test_roc_auc = scores['test_roc_auc'].mean()
    D = {'train_accuracy': train_accuracy,
         'test_accuracy': test_accuracy,
         'train_roc_auc': train_roc_auc,
         'test_roc_auc': test_roc_auc}
    return D


def print_results(model, scores):
    print()
    print(model)
    print("Accuracy score on train set:", scores['train_accuracy'])
    print("Accuracy score on test set:", scores['test_accuracy'])
    print("AUC score on train set:", scores['train_roc_auc'])
    print("AUC score on test set:", scores['test_roc_auc'])
    print()


if __name__ == '__main__':
    path = os.path.join('data', 'breast-cancer.csv')
    X, y = get_data(path)

    model = "Constant"
    clf = DummyClassifier(strategy='most_frequent')
    scores = get_scores(clf, X, y)
    print_results(model, scores)
    """
        Accuracy score on train set: 0.702
        Accuracy score on test set: 0.703
        AUC score on train set: 0.5
        AUC score on test set: 0.5
    """

    model = "Logistic Regression"
    clf = LogisticRegression(penalty='l2',
                             C=0.1,
                             solver='liblinear',
                             random_state=42)
    scores = get_scores(clf, X, y)
    print_results(model, scores)
    """
        Accuracy score on train set: 0.750
        Accuracy score on test set: 0.720
        AUC score on train set: 0.758
        AUC score on test set: 0.692
    """

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    model = "Neural Network"
    clf = MLPClassifier(hidden_layer_sizes=(10,),
                        activation='relu',
                        solver='lbfgs',
                        alpha=10.,
                        max_iter=200,
                        random_state=42)
    scores = get_scores(clf, X_norm, y)
    print_results(model, scores)
    """
        Accuracy score on train set: 0.860
        Accuracy score on test set: 0.731
        AUC score on train set: 0.917
        AUC score on test set: 0.662
    """
