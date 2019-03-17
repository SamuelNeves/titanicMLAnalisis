import pytest
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.svm import NuSVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
from pathlib import Path
import numpy as np

from src.pipeline.CategoricalEncoder import CategoricalEncoder
from src.pipeline.DataFrameSelector import DataFrameSelector
from src.pipeline.ValueSanitazer import ValueSanitizer
from src.pipeline.clf_switcher import ClfSwitcher
from sklearn.impute import SimpleImputer

from src.pipeline.debug import Debug


@pytest.fixture
def dataset():
    p = Path(__file__).resolve().parents[1]
    X = pd.read_csv(p.joinpath('data', 'interim', 'train.csv'))
    y = X['Survived']
    X.drop(columns='Survived', inplace=True)
    return X, y


@pytest.fixture
def train_test(dataset):
    x, y = dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=42)

    return x_train, x_test, y_train, y_test


@pytest.fixture
def pipeline():
    select_numeric_cols = FunctionTransformer(lambda X: X.select_dtypes(exclude=['object']), validate=False)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    numeric_features = ["SibSp", "Age", "Fare"]
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('debug', Debug()),

        ('scaler', StandardScaler())
    ])

    categorical_features = ["Ticket","Name","Embarked", "Sex", "Pclass","Cabin"]
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),

        ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    clf = Pipeline(steps=[

        ('preprocessor', preprocessor),
      ('clf', ClfSwitcher())

                          ])
    return clf



def test_hyper_parameter_optimization(train_test, pipeline):
    X_train, X_test, y_train, y_test = train_test
    parameters = [
        {
            'clf__estimator': [SGDClassifier()],  # SVM if hinge loss / logreg if log loss
            'clf__estimator__penalty': ('l2', 'elasticnet', 'l1'),
            'clf__estimator__max_iter': [50, 80],
            'clf__estimator__tol': [1e-4],
            'clf__estimator__loss': ['hinge', 'log', 'modified_huber'],
        },
        # {
        #     'clf__estimator': [MultinomialNB()],
        #     'clf__estimator__alpha': (1e-2, 1e-3, 1e-1),
        # },
        {
            'clf__estimator': [AdaBoostClassifier()],
            'clf__estimator__n_estimators': (10, 30, 50),
        },
        {
            'clf__estimator': [GradientBoostingClassifier()],
            'clf__estimator__loss': ('deviance', 'exponential'),
        },
        # {
        #     'clf__estimator': [NuSVC()],
        #     'clf__estimator__nu': ('deviance', 'exponential'),
        # }
    ]
    gscv = GridSearchCV(
        pipeline, parameters, cv=5, n_jobs=12, verbose=0,
    )
    # param optimization
    gscv.fit(X_train, y_train)
    score = gscv.score(X_test, y_test)
    print(score)
    assert isinstance(score, float)
