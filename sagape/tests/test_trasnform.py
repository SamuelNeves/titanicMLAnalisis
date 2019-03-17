import pandas as pd

import pytest
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from src.pipeline.transform import get_standard_scale_with_pca_etl


@pytest.fixture
def dataset():
    data = load_boston()
    return (
        pd.DataFrame(data['data'], columns=data.feature_names),
        pd.Series(data['target'], name='PRICE')
    )


@pytest.fixture
def sample(dataset):
    x, y = dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    return x_train, x_test, y_train, y_test


def test_transform_etl_with_ridge(sample):
    pipe = get_standard_scale_with_pca_etl()
    pipe.steps.append(('regressor', Ridge()))

    x_train, x_test, y_train, y_test = sample

    pipe = pipe.fit(x_train, y_train)
    score = pipe.score(x_test, y_test)
    print(score)
    assert isinstance(score, float)
    assert score > 0
