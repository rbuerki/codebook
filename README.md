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
_(The different transformations are demonstrated in nb-4 of the starbucks challenge.)
- `apply_log`: Transform values of selected columns to natural log. 
  NaN not affected by default, parameter can be changed.
- `apply_log10`: Transform values of selected columns to log10. 
  NaN not affected by default, parameter can be changed.
- `apply_box_cox`: Power transform values of selected columns with box-cox.
  NOTE: Cannot handle NaN and negvalues. Workaround to handle zero values.
- `apply_yeo_j`: Power transform values of selected columns with yeo-johnson.
  NOTE: Cannot handle NaN but yeo-johnson works on neg and zero values.

### EDA_functions.py - List of functions

*Distributions:*
- `plot_num_hist`: Display histograms for all numerical columns in DataFrame.
- `plot_num_box`: Display boxplots for all numerical columns in DataFrame.
- `plot_cat_pies`: Display pieplots for all categorical columns in DataFrame with 
  up to 30 unique values.

*Correlations:* 
- `plot_num_corrMap`: Display heatmap to show correlations between all numerical 
  columns in DataFrame.    
- `plot_corr_num_scatter`: Display scatterplots to visualize correlations between 
  all numerical features and target variable.
- `plot_num_corrBox`: Display boxplots to show correlations between all numerical 
  variables and target classes value in DataFrame.
- `plot_num_corrLine`: Display lineplots to show correlation details between all 
  numerical variables and target classes in DataFrame.
- `plot_cat_corrPoint`: Display pointplots (and corresponding piecharts) to show 
  correlations between all categorical columns and target classes in DataFrame.


### custom_transformers.py - List of classes

_(These classes were developed and used in the disaster message pipeline project.)
- `TypeSelector`: Selects columns from a DataFrame with specified datatype(s) 
  for further pipeline processing.  
- `CustomOneHotEncoder`: Custom OneHotEncoder based on Pandas get_dummies() 
  function. _(I prefer this over sk-learns built in OneHotEncoder because of the 
  possibility to define labels for the new dummy columns. This makes checking 
  for feature importance easier.)_ 


### hypothesis_functions.py - List of functions

_NOTE: The functions in this notebook work for calculations on PROPORTIONS only!
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