from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import pandas as pd


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        X_df_new = X_df.copy()
        X_df_new = compute_rolling_mean(X_df_new, 'Wrist_ACC_X', '5')
        return X_df_new


def compute_rolling_mean(data, feature, row_window):
    name = '_'.join([feature, row_window, 'mean'])
    data[name] = data[feature].rolling(int(row_window)).mean()
    data[name] = data[name].ffill().bfill()
    data[name].astype(data[feature].dtype)
    return data
