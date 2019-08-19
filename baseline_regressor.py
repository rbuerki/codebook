import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import resample  # for error calculation of feature weights

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
color = 'rebeccapurple'


class BaselineRegression:
    """Regression class for applying and evaluating
    different baseline regression models.
    """

    def __init__(self, df, target_col, model):
        """ Initialize BaselineRegression instance.

        Arguments:
        ----------
        - df: dataframe
        - target_col: target column for logistic prediction (as str)
        - model: sklearn model object including parameters
        """
        self._df = df.copy()
        self._target_col = target_col
        self._model = model

    def __repr__(self):
        """Function to output the characteristics of BaselineRegression
        instance. (Including the performance scores if the model is trained
        and evaluated.
        """

        try:
            return("{}\n\nRMSE on test data {:.2f}, r2-score {:.2f}."
                   .format(self._model, self._rmse, self._rsquared))
        except Exception:
            return(str(self._model))

    def go_quickDirty(self, tree_based, dummy_na=True, transform=True,
                      test_size=0.3, random_state=666):
        """ This function will call three hidden functions:
         1. Preprocess data according to model type (tree_based vs. other)
         2. Split, fit, predict
         3. Evaluate and output metrics

        Note: you can input any unprepared dataset, NaN values will be handled
        in a simple way, as will categorical values. Standard scaling is
        optional for non-tree based models, but there is no outlier treatment.
        Some of these operations actually should not be applied before splitting
        into test and training sets, but they are here for simplicity.

        Arguments:
        ----------
        - tree_based: bool, holding wheter preprocessing should be performed
            for tree-based models or other
        - dummy_na: bool, holding whether to dummy NA vals of categorical
            columns in own column or to ignore them (default=True).
            This argument is ignored if tree_based=True.
        - transform: bool, holding whether to normalize and scale the numerical
            features or to leave them as is (deflault=True)
            This argument is ignored if tree_based=True.
        - test_size: float, proportion of data to set aside in test dataset
            (default=0.3)
        - random_state: int, random state for splitting the data into training
            and test sets (default=666)

        Returns:
        --------
        - _rmse: Root Mean Squared Error of prediction
        - _rsquared: R squared / variance score (> 0.7 indicates good model)
        - _model: sklearn model object
        - _X_train, _X_test, _y_train, _y_test: output from sklearn train test
            split used for optimal model
        """

        self._tree_based = tree_based
        self._dummy_na = dummy_na
        self._test_size = test_size
        self._transform = transform
        self._random_state = random_state

        # Call hidden functions

        self.preprocess_NaN_cat_model()
        self.split_fit_predict_model()
        self.evaluate_model()

    def go_preprocessed(self, test_size=.3, random_state=666):
        """ This function is for datasets that have been manually preprocessed.
        It will:
        1. Split the data into an X matrix and a target vector y
        2. Create training and test sets of data
        3. Instantiate a LogisticRegression model with default parameters
        4. Fit the model to the training data
        5. Predict the target for the training data and the test data
        6. Obtain F1-score and ROC-AUC-Score

        Arguments:
        ----------
        - test_size: float, proportion of data to set aside in test dataset
            (default=0.3)
        - random_state: int, random state for splitting the data into training
            and test sets (default=666)

        Returns:
        --------
        - _rmse: Root Mean Squared Error of prediction
        - _rsquared: R squared / variance score (> 0.7 indicates good model)
        - _model: sklearn model object
        - _X_train, _X_test, _y_train, _y_test: output from sklearn train test
            split used for optimal model
        """

        self._test_size = test_size
        self._random_state = random_state

        # Call 'hidden' function
        self.split_fit_predict_model()
        self.evaluate_model()

    def preprocess_NaN_cat_model(self):
        """This 'hidden' function is called indirectly by go_quickDirty()
        and will:
        1. Drop the rows with missing target values
        2. Drop columns with NaN for all the values

        for non-tree-based models:
        3. Fill the mean of the column for any missing numerical values
        4. Standard-scale the numerical features (only if transform=True!)
        5. Use create_dummy_df to dummy categorical columns

        for tree-based models:
        3. Fill in the distinct value '-999' for any missing numeric values
        4. Label-encode the categorical columns
        """

        assert is_numeric_dtype(self._df[self._target_col]), \
                'target column must be numerical'

        # Clean rows with NaN in target col
        self._df = self._df.dropna(subset=[self._target_col], axis=0)
        # Drop columns with all NaN
        self._df = self._df.dropna(how='all', axis=1)

        if self._tree_based:
            # Impute distinct value for remaining missing values
            for col in self._df.select_dtypes(
                    include=['float', 'int']).columns:
                self._df[col] = self._df[col].fillna(-999)
            # OHE non-numerical columns and drop original columns
            for col in self._df.select_dtypes(
                    include=['object', 'category']).columns:
                self._df[col] = self._df[col].factorize()[0]

        else:
            # Impute mean for missing values in num cols
            for col in self._df.select_dtypes(include=['float', 'int']).columns:
                self._df[col].fillna(self._df[col].mean(), inplace=True)
                # Standard-scale numerical data if param transform=True
                if self._transform:
                    self._df[col] = scale(self._df[col])
            # OHE non-numerical columns and drop original columns
            for col in self._df.select_dtypes(
                    include=['object', 'category']).columns:
                self._df = pd.concat([self._df.drop(col, axis=1),
                    pd.get_dummies(self._df[col], prefix=col, prefix_sep='_',
                    drop_first=True, dummy_na=self._dummy_na)], axis=1)

    def split_fit_predict_model(self):
        """This 'hidden' function is called indirectly and will:
        1. Split the data into an X matrix and a target vector y
        2. Create training and test sets of data
        3. Instantiate a LogisticRegression model with default parameters
        4. Fit the model to the training data
        5. Predict the target for the training data and the test data
        """

        # Separate target column
        X = self._df.drop(self._target_col, axis=1)
        y = self._df[self._target_col].copy()

        # Split into train and test
        self._X_train, self._X_test, self._y_train, self._y_test = \
            train_test_split(X, y, test_size=self._test_size,
                             random_state=self._random_state)

        # Fit and predict the model
        self._model.fit(self._X_train, self._y_train)
        self._test_preds = self._model.predict(self._X_test)

    def evaluate_model(self):
        """Evaluate a machine learning model on four metrics:
        ROC AUC, precision score, recall score, and f1 score."""

        mse = mean_squared_error(self._y_test, self._test_preds)
        self._rmse = np.sqrt(mse)
        self._rsquared = r2_score(self._y_test, self._test_preds)

        # Print the metrics
        print(repr(self._model).split('(')[0])
        print("\nRMSE:", round(self._rmse, 4))
        print("r2:", round(self._rsquared, 4))

    def compare_to_naive(self, name="Naive Baseline"):
        """For a naive baseline, we can guess that has the a target value
        equal to the mean value of all instances in the training set. The
        metrics are the same as for the proper baseline model.
        """

        naive_guess = np.full(len(self._y_test), np.mean(self._y_train))

        print("Mean target value of training data is:",
              round(np.mean(self._y_train), 4))
        print("\n")

        naive_mse = mean_squared_error(self._y_test, naive_guess)
        naive_rmse = np.sqrt(naive_mse)
        naive_rsquared = r2_score(self._y_test, naive_guess)

        # Print the metrics
        print(name)
        print("\nRMSE:", round(naive_rmse, 4))
        print("r2:", round(naive_rsquared, 4))

    def plot_learning_curves(self, scoring='r2', n_folds=5):
        """
        Display learning curves based on n-fold cross validation.

        Arguments:
        ----------
        - scoring: str, evaluation score (default='r2')
        - n_folds: int, number of folds for cross validation (default=5)
        """

        # Apply sklearn learning_curve utility on X_train with n-fold CV
        N, trainCurve, valCurve = learning_curve(self._model,
            self._X_train, self._y_train, cv=n_folds,
            scoring=scoring, train_sizes=np.linspace(0.01, 1.0, 20),
            n_jobs=-1, verbose=1)

        # Calculate means and std deviation
        trainCurveMean = np.mean(trainCurve, axis=1)
        valCurveMean = np.mean(valCurve, axis=1)
        trainCurveStd = np.std(trainCurve, axis=1)
        valCurveStd = np.std(valCurve, axis=1)

        # Plot learning curves
        plt.figure(figsize=(16, 4))
        plt.plot(N, trainCurveMean, color=color,
                 marker='o', label='training score')
        plt.plot(N, valCurveMean, color='yellow', marker='o',
                 label='validation score')
        plt.fill_between(N, trainCurveMean - trainCurveStd, trainCurveMean +
                         trainCurveStd, alpha=0.2, color=color)
        plt.fill_between(N, valCurveMean - valCurveStd, valCurveMean +
                         valCurveStd, alpha=0.2, color='yellow')
        plt.hlines(np.mean([trainCurveMean[-1], valCurveMean[-1]]), N[0],
                   N[-1], color='black', linestyle='dashed')

        # Style plot
        # plt.ylim(0,1)
        plt.xlim(N[0], N[-1])
        plt.xlabel('training size')
        plt.ylabel(scoring)
        plt.title("Learning Curves {}".format(self._model))
        plt.legend(loc='lower right');

    def print_coef_weights(self, n_bootstrap=10):
        """ Output the estimates for coefficient weights and corresponding
        error. The error is calculated with help of bootstrap resamplings. This
        method may ouptut better results for models using L2 regularization
        (which is creating sparse feature sets).

        Arguments:
        ----------
        n_bootstrap: int, number of bootstrap resamplings (default=10)

        Returns:
        --------
        None, displays two dataframes, one with with coef weights for all
        features separately, and one with cumulated weights for the categorical
        features.
        """

        assert self._tree_based is False, "works for linear models only"

        self._coef = self._model.coef_[0]
        coef_df = pd.DataFrame(index=self._X_train.columns)
        coef_df['effect'] = self._coef.round(1)

        # Calculate modulo of coef just for sorting the df, then drop
        coef_df['abs_coef'] = np.abs(coef_df['effect'])
        coef_df = coef_df.sort_values('abs_coef', ascending=False)
        coef_df.drop('abs_coef', axis=1, inplace=True)

        # Add uncertainty measure with help of bootstrap resamling
        np.random.seed(1)
        err = np.std([self._model.fit(*resample(self._X_train, self._y_train))
                     .coef_[0] for i in range(n_bootstrap)], 0)
        coef_df['error'] = err.round(0)

        display(coef_df)
        print('\n')

        # Create df with cumulative weights for categorical features
        coef_df['category'] = coef_df.index.str.split('_').str.get(0)
        # coef_df['sub-category'] = coef_df[1].str.split('-').str.get(1)
        coef_df_cum = coef_df.groupby('category').sum().sort_values('effect',
                ascending=False)

        coef_df_cum['abs_coef'] = np.abs(coef_df_cum['effect'])
        coef_df_cum = coef_df_cum.sort_values('abs_coef', ascending=False)
        coef_df_cum.drop('abs_coef', axis=1, inplace=True)

        display(coef_df_cum)

    def plot_feature_weights(self, max_cols=None):

        assert self._tree_based, "works for tree-based models only"

        self._weights = self._model.feature_importances_
        self._weights_named_df = pd.DataFrame(sorted(zip(self._weights,
                                                         self._X_train.columns),
                                              reverse=True))

        if max_cols is not None:
            self._weights_named_df = self._weights_named_df.iloc[:, :max_cols]
        else:
            plt.figure(figsize=(16, 5))
            plt.bar(np.arange(len(self._weights_named_df)),
                    self._weights_named_df[0],
                    width=0.5,
                    align="center",
                    color='yellow',
                    label="Feature Weight")
            plt.bar(np.arange(len(self._weights_named_df)) - 0.3,
                    np.cumsum(self._weights_named_df[0]),
                    width=0.4,
                    align="center",
                    color=color,
                    label="Cumulative Feature Weights")
            plt.title("Normalized Weights for Predictive Features / Attributes")
            plt.ylabel("Weights")
            plt.xlabel("Features / Attributes")
            plt.xticks(np.arange(len(self._weights_named_df)),
                       self._weights_named_df[1],
                       rotation=90)
            plt.legend(loc='upper left');
