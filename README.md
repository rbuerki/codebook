#  Codebook

Some modules containing collections of functions and classes written for reusability in my projects.

## EDA.py - List of Functions
(last update Nov 2020)

see notebook in demo folder

Dataframe Values:
- `display_distinct_values`: Return a dataframe containing the number
   of distinct values for each column of the input dataframe.
- `display_value_counts`: Display a dataframe containing the value
   counts and their respective pct for a column or a list of columns.
- `display_tail_transposed`: Return transposed tail of the passed
   dataframe with cols shown as rows and values for 5 instances as cols.
- `display_dtypes`: Return a dataframe showing the count of different
   datatypes for the columns in the input dataframe.

Missing Values and Duplicates:
- `display_nan`: Return a dataframe showing the missing values with
   their respective percentage of the total values in a column.
- `plot_nan`: Display a heatmap of the input dataframe, highlighting
   the missing values.
- `display_duplicates`: Print a summary of the column-wise duplicates
   in the input dataframe.

Distributions:
- `plot_distr_histograms`: Display a histogram for every numeric
   column in the input dataframe.
- `plot_distr_boxplots`: Display a boxplot for every numeric
   column in the input dataframe.
- `plot_distr_pies`: Display a pieplot for every column of dtype
  "category" (with up to 30 distinct values) in the input dataframe.
- `plot_distr_pdf_ecdf`: Display a histogram overlaid with an ECDF 
   for every numeric column in the input dataframe.

Correlations:
- `plot_corr_full_heatmap`: Display a heatmap to show the correlations
   between all numeric columns in the Dataframe.
- `plot_corr_to_target_barchart`: Display a barchart for every numeric
   feature in the input dataframe to show the correlation to a
   numeric target variable.
- `plot_corr_to_target_regplots`: Display a regplot for every numeric
   feature in the input dataframe to show the correlation to a
   numeric target variable.
- `plot_corr_to_target_lineplots`: Display a lineplot for every numeric
  feature in the input dataframe with up to (by default) 100 distinct
  values to analyze the correlation to a numeric target variable.
- `plot_corr_to_target_boxplots`: Display a boxplot for every numeric
   feature column in the input dataframe to analyze the correlation to
   a target variable with few distinct values (any dtype possible).
- `plot_corr_to_target_pointplots_with_pies`: Display a pointplot
   (and corresponding piechart) for every feature with dtype "category"
   in the input dataframe to display the correlation to a numeric
   target variable.
- `plot_corr_to_target_stripplots`: Display a stripplot for each
   feature with dtype "category" in the input dataframe to analyze
   the correlation to a numeric target variable.

Cumulative Sums / Counts:
- `display_cumcurve_stats`: Return a dataframe with cumsum stats for an
  iterable of numeric values.
- plot_cumsum_curve`: Display a cumsum curve for an iterable of numeric
  values.

## clean.py - List of Functions
(last update Nov 2020)

see notebook in demo folder

Columns:
- `prettify_column_names`: Replace whitespace in column labels with
   an underscore and, by default, change to all lowercase.
- `delete_columns`: Delete selected columns permanently from the
   input dataframe.

Outliers:
- `count_outliers_IQR_method`: Detect outliers in specified columns
   depending on specified distance from 1th / 3rd quartile. NaN ignored.
- `remove_outliers_IQR_method`: Remove outliers in specified columns
   depending on specified distance from 1th / 3rd quartile. NaN ignored.
- `winsorize_values`: Return a winsorized version of the selected
   columns.

Transformations:
- `transfrom_data`: Apply the desired transformation on the selected
   columns. (Methods are log, log10, box-cox or yeo-johnson.)

--<!-- markdownlint-capture -->

## Older Stuff

### custom_transformers.py - List of Classes
(last update: long ago ...)

- `ColumnSelector`: Selects  the defined  columns from a DataFrame for further
    processing. Makes sure, that only the these columns are processed. Valuable
    in automated production settings (e.g. ETL microservice) or if you want to
    experiment with different feature settings.
- `TypeSelector`: Selects columns from a DataFrame with specified datatype(s) for
    further parallelized pipeline processing  with _FeatureUnion_. (It is of no
    use when working with ColumnTransformer.)
- `PassthroughTransformer`: Can be used within _ColumnTransformer_. Just passes the
    data on as it is. Can be used when you want to get the feature names for the
    transformed dataframe as the built in 'passthrough' argument in sklearn 0.20
    does not (yet) support get_feature_names(). See [here](https://stackoverflow.com/questions/53382322/adding-get-feature-names-to-columntransformer-pipeline) for background info.


### hypothesis_functions.py - List of Functions
(last update: long ago ...)

_NOTE: The functions in this notebook are for calculations on PROPORTIONS only!
(They have been developed in the small projects in the experimental design repository.)_
- `calc_confidence_bounds_binomial`: Compute lower and upper bounds for a defined
  confidence level based on a random variable with binomial / normal distribution.
- `calc_experiment_size`: Compute minimum number of samples for each group needed
  to achieve a desired power level for a given effect size.
- `calc_invariant_population`: Compute statistics to check if your random
  population sizing is within the expected standard error for a 50/50 split.
- `calc_experiment_results`: Compute observed difference with it's lower and upper
  bounds based on a defined conficence level.


### baseline_regressor.py
(last update: long ago ...)

`BaselineRegression`: Regression class based on sklearn for applying and evaluating
different models. Needs a sklearn regression model object as input.

_Basic functions:_
- `go_quickDirty`: Apply regression modelling to unprepared / dirty data. Preprocessing
    varies for tree-based and non-tree-based models.
- `go_preprocessed`: Apply regression modelling to properly prepared data.

_Evaluation:_
- `plot_learning_curves`: Display learning curves based on n-fold cross validation.
- `print_coef_weights`: Output estimates for coefficient weights and corresponding
      error for estimatores that have a coeff attribute (linear models)
- `print_feature_weights`: Outputs feature weights for estimators that have a
    get_feature_weights() method (tree-based models)


### baseline_classifier.py
(last update: long ago ...)

`BaselineClassification`: Classification class based on sklearn for applying and
evaluating different models. Needs a sklearn classification model object as input.

_Basic functions:_
- `go_quickDirty`: Apply classification modelling to unprepared / dirty data. Preprocessing
    varies for tree-based and non-tree-based models.
- `go_preprocessed`: Apply classification modelling to properly prepared data.

_Evaluation:
- `compare_to_naive:` Compute metrics for a naive baseline as a comparison to
    the results of our not so naive baseline.
- `plot_learning_curves`: Display learning curves based on n-fold cross validation.
- `plot_ROC_curve`: Plot area under the ROC-curve (ROC-AUC).
- `print_classification_report`: ... name says it all ... print simple classification
    report for evaluated model.
- `plot_confusion matrix`: Plot confusion matrix and detailed metrics for
    evaluated model.
- `print_coef_weights`: Output estimates for coefficient weights and corresponding
    error for estimatores that have a coeff attribute (linear models)
- `print_feature_weights`: Outputs feature weights for estimators that have a
    get_feature_weights() method (tree-based models)


## Requirements

These functions require **Python 3.6** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [seaborn](http://seaborn.org)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [tqdm](https://pypi.org/project/tqdm/)