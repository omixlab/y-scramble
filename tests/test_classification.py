import sys
sys.path.append(".")
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from y_scramble import Scrambler

def test_iris_train_test_split():

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = DecisionTreeClassifier()
    scrambler = Scrambler(model=model)
    scrambler.validate(X_train, X_test, y_train, y_test, trained=False, scoring="f1_weighted")

