import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, \
    precision_score, recall_score, confusion_matrix, classification_report
from sklearn.utils import resample  # for error calculation of feature weights

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
color = 'rebeccapurple'


class LogRegModel:
    """ Logistic Regression class for applying and evaluating
    different logistic regression models.
    """

    def __init__(self, df, target_col, model):
        """ Initialize logRegModel instance.

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
        """Function to output the characteristics of logRegModel instance.
        Including the performance scores if the model is trained and evaluated.
        """

        try:
            return("{}\n\nF1-score on test data {:.2f}, AUC-score {:.2f}."
                   .format(self._model, self._f1_score, self._auc))
        except Exception:
            return(str(self._model))

    def go_quickDirty(self, dummy_na=True, test_size=0.3, random_state=666):
        """ This function will:
        1. Drop the rows with missing target values
        2. Drop columns with NaN for all the values
        3. Use create_dummy_df to dummy categorical columns
        4. Fill the mean of the column for any missing values
        5. Split the data into an X matrix and a target vector y
        6. Create training and test sets of data
        7. Instantiate the model
        8. Fit the model to the training data
        9. Predict the target for the training data and the test data
        10. Obtain F1-score and ROC-AUC-score

        Note: you can input any unprepared dataset, NA values will be handled in a simple way, ALL NON NUMERIC variables dummied. But there
        is NO OUTLIER TREATMENT OR SCALING. So you better make sure
        to set the parameter 'normalize=True' for your model input. (???)

        Arguments:
        ----------
        - dummy_na: bool, holding whether to dummy NA vals of categorical
            columns in own column or to ignore them (default=True)
        - test_size: float, proportion of data to set aside in test dataset
            (default=0.3)
        - random_state: int, random state for splitting the data into training
            and test sets (default=666)

        Returns:
        --------
        - f1_score: ...
        - auc_score: ROC-AUC score
        - logreg_model: sklearn model object
        - X_train, X_test, y_train, y_test: output from sklearn train test
            split used for optimal model
        """

        self._dummy_na = dummy_na
        self._test_size = test_size
        self._random_state = random_state

        # Call hidden functions
        self.preprocess_NaN_cat_model()
        self.split_fit_predict_model()
        self.evaluate_model()

    def go_preprocessed(self, test_size=.3, random_state=666):
        """ This function will:
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
        - f1_score: ...
        - auc_score: ROC-AUC score
        - logreg_model: sklearn model object
        - X_train, X_test, y_train, y_test: output from sklearn train test
            split used for optimal model
        """

        self._test_size = test_size
        self._random_state = random_state

        # Call 'hidden' function
        self.split_fit_predict_model()
        self.evaluate_model()

    def preprocess_NaN_cat_model(self):
        """This 'hidden' function is called indirectly and will:
        1. Drop the rows with missing target values
        2. Drop columns with NaN for all the values
        3. Use create_dummy_df to dummy categorical columns
        4. Fill the mean of the column for any missing values
        """

        assert self._df[self._target_col].dtype == 'int64' or \
            self._df[self._target_col].dtype == 'float64', \
            'target column must be numerical'

        # Clean rows with NaN in target col
        self._df = self._df.dropna(subset=[self._target_col], axis=0)
        # Drop columns with all NaN
        self._df = self._df.dropna(how='all', axis=1)
        # Impute mean for missing values in num cols
        for col in self._df.select_dtypes(include=['float', 'int']).columns:
            self._df[col].fillna(self._df[col].mean(), inplace=True)
        # OHE non-numerical columns and drop original columns
        for col in self._df.select_dtypes(
                include=['object', 'category']).columns:
            self._df = pd.concat([self._df.drop(col, axis=1), \
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
        self._test_probs = self._model.predict_proba(self._X_test)

    def evaluate_model(self):
        """Evaluate a machine learning model on four metrics:
        ROC AUC, precision score, recall score, and f1 score."""

        self._f1_score = f1_score(self._y_test, self._test_preds)
        self._auc = roc_auc_score(self._y_test, self._test_probs[:, 1])  # !

        # Print the metrics
        print(repr(self._model).split('(')[0])
        print("\nROC AUC:", round(self._auc, 4))
        # Iterate through remaining metrics, use .__name__ attribute
        for metric in [precision_score, recall_score, f1_score]:
            print("{}: {}".format(metric.__name__,
                  round(metric(self._y_test, self._test_preds), 4)))

    def compare_to_naive(self, name="Naive Baseline"):
        """For a naive baseline, we can randomly guess that an instance is of
        the positive class in the same frequence of the positive classified
        instances in the training data. We'll assess the predictions using the
        same metrics as the proper baseline model.
        """

        np.random.seed(self._random_state)
        naive_guess = np.random.binomial(1, p=np.mean(self._y_train),
                                         size=len(self._y_test))

        print("Percentage of positive class in training data is:",
              round(np.mean(self._y_train), 4))
        print("\n")

        # Print the metrics

        print(name)
        print("\nROC AUC:", roc_auc_score(self._y_test,
              np.repeat(np.mean(self._y_train), len(self._y_test))))

        # Iterate through remaining metrics, use .__name__ attribute
        for metric in [precision_score, recall_score, f1_score]:
            print("{}: {}".format(metric.__name__,
                  round(metric(self._y_test, naive_guess), 4)))

    def plot_learning_curves(self, scoring='neg_log_loss', n_folds=5):
        """
        Display learning curves based on n-fold cross validation.

        Arguments:
        ----------
        - scoring: str, evaluation score (default='neg_log_loss')
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

    def plot_ROC_curve(self):
        """Plot area under the ROC-curve (ROC-AUC)."""

        # Calculate ROC-curve
        fpr, tpr, thresholds = roc_curve(self._y_test, self._test_probs[:, 1])

        # Plot ROC curve
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, label="{} (area = {:.3f})".format("ROC AUC:",
                 self._auc), color=color)
        plt.fill_between(fpr, tpr, alpha=0.2, color=color)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate (1 - Specificity)', size=12)
        plt.ylabel('True Positive Rate (Sensitivity)', size=12)
        plt.title('ROC - Aera under the curve', size=12)
        plt.legend(loc='lower right');

    def print_classification_report(self):
        """Print classification report for evaluated model."""

        print(classification_report(self._y_test, self._test_preds))

    def print_confusion_matrix(self):
        """Print and confusion matrix and detailed metrics for evaluated model.
        """
        conf_matrix = confusion_matrix(self._y_test, self._test_preds)
        tn, fp, fn, tp = conf_matrix.ravel()

        print(conf_matrix)
        print("\n")
        for value, name in {tp: "True positives",
                            fp: "False positives",
                            tn: "True negatives",
                            fn: "False negatives",
                            }.items():
            print("{}: {} ({:.2f}%)".format(name, value,
                                            value / conf_matrix.sum() * 100))
        print("\nProportion of misclassified instances in total:",
              ((fp+fn) / (tp+fp+tn+fn)))
        print("Proportion of misclassified positives:", (fp/(fp+tp)))
        print("Proportion of misclassified negatives:", (fn/(fn+tn)))

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
        coef_df: dataframe, holding estimate for coeff weights and error
        """

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

        return coef_df
