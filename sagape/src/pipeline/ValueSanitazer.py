from sklearn.base import TransformerMixin


class ValueSanitizer(TransformerMixin):

    def __init__(self, vocabulary_dict):
        if type(vocabulary_dict) is not dict:
            raise ValueError("Input value for mapping is not a dict")
        self.vocabulary_dict = vocabulary_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.vocabulary_dict[item] if item in self.vocabulary_dict else item for item in X]
