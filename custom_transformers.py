
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects  the defined  columns from a DataFrame for further processing. 
    Makes sure, that only the these columns are processed. Valuable in automated 
    production settings (e.g. ETL microservice) or if you want to experiment
    with different feature settings.

    Arguments:
    ----------
    - columns: list of strings, containting the column names of the columns
        to be selected

    Returns:
    --------
    - X: DataFrame, sub-selection of X including the defined columns.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame), "input must be DataFrame"

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("DataFrame does not include: {}".format(cols_error))
            


class TypeSelector(BaseEstimator, TransformerMixin):
    """Selects columns from a DataFrame with specified datatype(s) for further 
    pipeline processing  with FeatureUnion. (Is no use when working with 
    ColumnTransformer.)

    Arguments:
    ----------
        dtype = dtype to be selected

    Returns:
    --------
    - X: DataFrame, sub-selection of X  including columns with the 
        selected dtype.
    """

    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame), "input must be DataFrame"
        
        return X.select_dtypes(include=[self.dtype])

    def get_feature_names(self):
        return self.X.select_dtypes(include=[self.dtype]).columns.tolist()



class PassthroughTransformer(BaseEstimator, TransformerMixin):
    """A custom Passtrough Transformer that can be used with ColumnTransformer. 
    It just passes the data as is. Can be used when you want to get the 
    feature names for the transformed dataframe as the built in 'passthrough' 
    argument does not yet support get_feature_names(). 

    See here for background information: 
    https://stackoverflow.com/questions/53382322/adding-get-feature-names-to-columntransformer-pipeline
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.X = X
        return X

    def get_feature_names(self):
        return self.X.columns.tolist()