import sys
sys.path.append(".")
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from y_scramble import Scrambler

def test_iris_train_test_split():
    X, y = load_iris(return_X_y=True)
    model = DecisionTreeClassifier()
    scrambler = Scrambler(model=model)
    return (scrambler.validate(X, y, method="train_test_split", scoring="accuracy"))

def test_iris_cross_validation():
    X, y = load_iris(return_X_y=True)
    model = DecisionTreeClassifier()
    scrambler = Scrambler(model=model)
    return (scrambler.validate(X, y, method="cross_validation", scoring="accuracy", cv_kfolds=20))