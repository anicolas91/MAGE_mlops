import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from typing import Tuple
from sklearn.base import BaseEstimator
from pandas import Series
from scipy.sparse._csr import csr_matrix

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def transform(df: pd.DataFrame, **kwargs) -> Tuple[DictVectorizer, BaseEstimator]: #, csr_matrix, Series]:
    # vectorize features
    dv = DictVectorizer()

    categorical = ['PULocationID', 'DOLocationID'] # pickup and dropoff location
    numerical = ['trip_distance'] # distance of trip

    train_dicts = df[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    # fit a linear regression model
    target = 'duration'
    y_train = df[target].values

    model = LinearRegression()
    model.fit(X_train,y_train)

    # print out intercept
    print('intercept:')
    print(model.intercept_)

    return dv, model #, X_train, y_train



