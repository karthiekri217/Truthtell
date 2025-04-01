# models.py
import numpy as np

class EnsembleModel:
    def __init__(self, models):
        """
        models: a dictionary of base models, e.g. {'lr': LogisticRegression(), ...}
        """
        self.models = models
    
    def fit(self, X, y):
        for model in self.models.values():
            model.fit(X, y)
    
    def predict_proba(self, X):
        predictions = np.zeros((X.shape[0], 2))
        for model in self.models.values():
            predictions += model.predict_proba(X)
        return predictions / len(self.models)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
