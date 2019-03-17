
import typing as tp
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, ClusterMixin

CLF = tp.Union[ClassifierMixin, ClusterMixin, RegressorMixin]

class ClfSwitcher(BaseEstimator):
    def __init__(self, estimator: CLF=None):
        self.estimator = estimator

    def fit(self, x, y=None):
        self.estimator.fit(x,y)
        return self

    def predict(self, x):
        return self.estimator.predict(x)

    def predict_proba(self, x):
        return self.estimator.predict_proba(x)

    def score(self, x, y, sample_weight=None):
        return self.estimator.score(x,y, sample_weight)

