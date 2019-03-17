from sklearn.base import BaseEstimator, TransformerMixin

from prettytable import PrettyTable
class Debug(BaseEstimator, TransformerMixin):

    def transform(self, X):
        print(X[0:10][:])
        print('---------')
        # print(X.head(4))
        self.shape = X.shape
        # what other output you want
        return X

    def fit(self, X, y=None, **fit_params):
        return self