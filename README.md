#  Codebook

Small collections of functions and classes written for reusability in my projects.

### cleaning_functions.py - List of functions

*Columns:*
- `edit_column_names`: Replace whitespace in column labels with underscore
  and change to all lowercase (optional).
- `change_dtypes`: Change different datatypes for selected columns.
- `delete_columns`: Delete columns permanently from a dataframe.

*Missing Values:*
- `plot_NaN`: Plot heatmap with all NaN in DataFrame.
- `list_NaN`: List columns with missing values and respective count of NaN.
- `handle_NaN`: Apply different strategies for NaN handling in selected
  columns (simplistic approach).

*Duplicates:*
- `list_duplicates`: Display the columns containing column-wise duplicates.

*Outliers:*
- `count_outliers_IQR_method`: Detect outliers in specified columns 
  depending on specified distance from 1th / 3rd quartile. NaN ignored.
- `remove_outliers_IQR_method`: Remove outliers in specified columns 
  depending on specified distance from 1th / 3rd quartile. NaN ignored.

*Transformations:*
_(The different transformations are demonstrated in nb-4 of the starbucks challenge.)_
- `apply_log`: Transform values of selected columns to natural log. 
  NaN not affected by default, parameter can be changed.
- `apply_log10`: Transform values of selected columns to log10. 
  NaN not affected by default, parameter can be changed.
- `apply_box_cox`: Power transform values of selected columns with box-cox.
  NOTE: Cannot handle NaN and negvalues. Workaround to handle zero values.
- `apply_yeo_j`: Power transform values of selected columns with yeo-johnson.
  NOTE: Cannot handle NaN but yeo-johnson works on neg and zero values.

### EDA_functions.py - List of functions

*General*
- `display_tail_transposed`: Display transposed tail of DataFrame with all the 
  features as rows and values for 5 instances as columns.

*Distributions:*
- `plot_num_hist`: Display histograms for all numerical columns in DataFrame.
- `plot_num_box`: Display boxplots for all numerical columns in DataFrame.
- `plot_cat_pies`: Display pieplots for all categorical columns in DataFrame with 
  up to 30 unique values.

*Correlations:* 
- `plot_num_corrMap`: Display heatmap to show correlations between all numerical 
  columns in the Dataframe.    
- `plot_corr_bar_num_target`: Display sorted barchart to show correlations between 
  all numerical features and numerical target variable.
- `plot_corr_scatter_num_target`: Display scatterplots to visualize correlations 
  between all numerical features and numerical target variable.
- `plot_corr_box_num_target`: Display boxplots to show correlations between all 
  numerical features and target classes.
- `plot_corr_line_num_target`: Display lineplots to show correlation details 
  between all numerical features and target classes.
- `plot_corr_strip_cat_target`: Display stripplots to show correlations between 
  the categorical features and numerical target variable.
- `plot_corr_point_cat_target`: Display pointplots (and corresponding piecharts) 
  to show correlations between all categorical columns and target classes.


### custom_transformers.py - List of classes


- `ColumnSelector`: Selects  the defined  columns from a DataFrame for further 
    processing. Makes sure, that only the these columns are processed. Valuable 
    in automated production settings (e.g. ETL microservice) or if you want to 
    experiment with different feature settings.
- `TypeSelector`: Selects columns from a DataFrame with specified datatype(s) for 
    further parallelized pipeline processing  with _FeatureUnion_. (It is of no 
    use when working with ColumnTransformer.)  
- `CustomOneHotEncoder`: Can be used within _ColumnTransformer_. Just passes the 
    data on as it is. Can be used when you want to get the feature names for the 
    transformed dataframe as the built in 'passthrough' argument in sklearn 0.20  
    does not (yet) support get_feature_names(). See [here](https://stackoverflow.com/questions/53382322/adding-get-feature-names-to-columntransformer-pipeline) for background info.


### hypothesis_functions.py - List of functions

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


### linRegModel_class.py 

`linRegModel`: Linear Regression class based on sklearn for applying and evaluating 
linear models. Needs a sklearn linReg model object as input.

_Basic functions:_  
- `go_quickDirty`: apply linear regression to unprepared / dirty data
- `go_preprocessed`: apply linear regression to properly prepared data

_Evaluation:_
- `plot_learning_curves`: Display learning curves based on n-fold cross validation.
- `print_coef_weights`: Output estimates for coefficient weights and corresponding 
      error. The error is calculated using bootstrap resamplings of the data.


## Install

These functions require **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [seaborn](http://seaborn.org)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [tqdm](https://pypi.org/project/tqdm/)