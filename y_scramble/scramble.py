from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import SCORERS
import pandas as pd
import numpy as np
import scipy
import copy
import json

class Scrambler:

    def __init__(self, model, iterations=100):
        self.base_model = model
        self.scrambled_models = []
        self.iterations = iterations
    
    def validate(self, X_train=None, X_test=None, y_train=None, y_test=None, scoring="accuracy", trained=True, pvalue_threshold=0.05):
        """
        Run a y-
        """
        scrambled_model_scores = []
        model_scorer = SCORERS.get(scoring)

        if model_scorer is None:
            raise Exception(f"scoring function '{model_scorer}' is not a sklearn scorer")
        
        if not trained:
            self.base_model.fit(X_train, y_train)

        y_pred = self.base_model.predict(X_test)
        base_model_score = model_scorer(self.base_model, X_test, y_test)

        for _ in range(self.iterations):

            np.random.shuffle(X_train)
            self.base_model.fit(X_train, y_train)
            
            scrambled_model_scores.append(model_scorer(self.base_model, X_test, y_test))

        all_scores = [base_model_score, *scrambled_model_scores]

        all_scores_zscores = scipy.stats.zscore(all_scores)
        all_scores_pvalues = scipy.stats.norm.sf(abs(all_scores_zscores))*2

        all_scores_significances = all_scores_pvalues <= pvalue_threshold

        results = {
            'base_model': {
                'score': all_scores[0], 
                'p-value': all_scores_pvalues[0], 
                'z-score': all_scores_zscores[0], 
                'significancy': all_scores_significances[0] 
            },
            'scrambled_model': {
                'scores': all_scores[1::], 
                'p-values': all_scores_pvalues[1::], 
                'z-scores': all_scores_zscores[1::], 
                'significances': all_scores_significances[1::]
            }
        }

        return results