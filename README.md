# Y-Scramble

Y-Scramble is a simple python package to perform y-randomization validation
of machine learning models. It can be used for classification and regression tasks
and accepts models following the `scikit-learn`interface, and the user may use all *scorers* available at `scikit-learn` (`accuracy`, `recall`, `precision`).

## Installing

Y-Scramble can be installed from PyPI using the following command:

```
$ pip install y-scamble
```

## Usage

```python

from y_scramble import Scrambler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = DecisionTreeClassifier()

scrambler = Scrambler(model=model, iterations=1000)

scores, zscores, pvalues, significances = scrambler.validate(
    X, y, 
    scoring="accuracy", 
    cross_val_score_aggregator="mean", 
    pvalue_threshold=0.01
)
```

The `scramble` object returns the scores, z-scores, p-values and the significancy
information for the model trained (`base_model`) using the default dataset and for different randomized versions as well (`scrambled_models`). These results are stores in numpy arrays, where the position of index 0 represents the `base_model`and the others the `scrambled_models`.

The score of the `base_model` is stored in `scores[0]`, and i's p-values is stored in
`pvalues[0]`. If this p-value is significant, the value of `significances[0]` will be
`True`, indicating that `base_model` shows a significantly better result when comparing to the randomized models. Following the same logic, `scores[1]` to `scores[1000]`, for example, will store the score values for the randomized model `1` and `1000`, respectively.



