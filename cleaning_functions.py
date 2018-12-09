"""
LIST OF FUNCTIONS

Columns
- edit_column_names: Replace whitespace in column labels with underscore
  and change to all lowercase (optional).
- change_dtypes: Change different datatyes for selected columns.
- delete_columns: Delete columns permanently from a dataframe.

Missing Values
- plot_NaN: Plot heatmap with all NaN in DataFrame.
- list_NaN: List columns with missing values and respective count of NaN.
- handle_NaN: Apply different strategies for NaN handling in selected
  columns (simplistic approach).

Duplicates:
- list_duplicates: Display the columns / containing column-wise duplicates.

Outlier Removal
- count_outliers_IQR_method: Detect outliers in specified columns 
  depending on specified distance from 1th / 3rd quartile. NaN ignored.
- remove_outliers_IQR_method: Remove outliers in specified columns 
  depending on specified distance from 1th / 3rd quartile. NaN ignored.

Transformations
- apply_log10: Transform values of selected columns to Log10. 
  NaN not affected by default, parameter can be changed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


### COLUMNS - Datatypes and Removal

def edit_column_names(df, lowercase=True):
    """Replace whitespace in column labels with underscore and change
       to all lowercase (this param can be changed).
    Params
    =====
        df: DataFrame
        lowercase: changes column labels to all lowercase, default True
    """
    if lowercase:
        df.columns = (col.lower().replace(' ', '_') for col in df.columns)
    else:
        df.columns = (col.replace(' ', '_') for col in df.columns)

    return df
   
        
def count_dtypes(df):
    dtypes_dict = {'numerical' : tuple(df.select_dtypes(include = ['float64', 'int64', 'float32', 'int32'])),
                   'categorical' : tuple(df.select_dtypes(include = ['category'])),
                   'object' : tuple(df.select_dtypes(include = ['object'])),
                   'datetime' : tuple(df.select_dtypes(include = ['datetime'])),
                   }
    counter = 0
    print("Total number of columns: {}".format(df.shape[1]))
    for datatype, cols_list in dtypes_dict.items():
        if len(cols_list) >0:
            print("- Columns with dtype {}: {}".format(datatype, len(cols_list)))
            counter += len(cols_list)
            
    #safety check: # if assert breaks some exotic dtype may have been missed
    assert counter == df.shape[1]  
    

def change_dtypes(df, cols_to_category=[], cols_to_object=[], cols_to_integer=[], 
                  cols_to_float=[], cols_to_datetime=[], datetime_pattern="%Y/%m/%d"):
    """Transform datatyes of selected columns in a dataframe.
       Return transformed DataFrame.
       Params
       ======
           df: DataFrame
           cols_to_category: list of colums to category, default empty
           cols_to_object: list of colums to string, default empty
           cols_to_integer: list of colums to integer, default empty
           cols_to_float: list of colums to float, default empty
           cols_to_datetime: list of colums to datetime, default empty
           datetime_pattern: datetime pattern, default = "%Y/%m/%d"
    """
    dtypes_dict = {tuple(cols_to_category) : 'category', 
                   tuple(cols_to_object) : str, 
                   tuple(cols_to_integer) : np.int64, 
                   tuple(cols_to_float) : np.float64,
                   }
    
    for cols_list, datatype in dtypes_dict.items():
        if cols_list != None:
            for col in cols_list:
                if col in df.columns:
                    df[col] = df[col].astype(datatype)
                else:
                    display(col + " not found")
 
    # different handling for datetime columns
    if cols_to_datetime != None:
        for col in cols_to_datetime:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format=datetime_pattern)
            else:
                display(col + " not found")

    return df


def delete_columns(df, cols_to_delete):
    """Delete columns permanently from a dataframe.
        Params
        ======
            df: DataFrame
            cols_to_delete: list of columns to delete
    """   
    for col in cols_to_delete:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            display(col + " successfully deleted")


### MISSING VALUES - Detection and handling

def plot_NaN(df, figsize=(18, 6), cmap='viridis'):
    """Display heatmap of DataFrame with NaN.
    Params
        ======
            df: DataFrame
            figsize: default is (18, 6)
            cmap: default is 'viridis'
    """
    plt.figure(figsize=figsize)
    sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis');


def list_NaN(df):
    """Display the columns with missing values and respective number of NaN.
    Params
        ======
            df: DataFrame
    """
    print("Number of NaN per column:")
    for col in df:
        nan_count = df[col].isnull().sum()
        if nan_count > 0:
            print("{}: {} ({:.2f}%)".format(
                df[col].name, nan_count, nan_count/len(df)))
            

def handle_NaN(df, cols_to_impute_num=None, cols_to_impute_cat=None, 
               cols_to_drop=None, drop_all_NaN=False):
    """Handle NaN with different strategies for selected columns. 
       Return transformed DataFrame. Note: Use with caution, there are 
       more sophisticated solutions around.
    
    ARGUMENTS:
        df: DataFrame
        cols_to_impute_num: list of num columns to impute median (default: None)
        cols_to_impute_cat: list of categorical columns to impute mode (default: None)
        cols_to_drop: list of columns to drop entirely (default: None)
        drop_all_NaN: if True ALL remaining rows with NaN will be removed, (default: False)
    """
    if cols_to_impute_num != None:    
        for col in cols_to_impute_num:
            if col in df.columns:
                display("{} - median value to impute: {}".format(
                    col, df[col].median()))
                df[col] = df[col].fillna(df[col].median())
            else:
                display(col + " not found")
    if cols_to_impute_cat != None:
        for col in cols_to_impute_cat:
            if col in df.columns:
                display("{} - most frequent value to impute: {}".format(
                    col, df[col].value_counts().index[0]))
                df[col] = df[col].fillna(df[col].value_counts().index[0])
            else:
                display(col + " not found")
    if cols_to_drop != None:
        for col in cols_to_drop:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
            else:
                display(col + " not found")
    if drop_all_NaN:
        df = df.dropna(how='any')   # drop remaining rows with any NaNs       
    return df
 

 ### DUPLICATES


def list_duplicates(df):
    """Display the columns / containing column-wise duplicates.
    
    ARGUMENTS:
        df: DataFrame
    """
    print("Number of column-wise duplicates per column:")
    for col in df:
        dup = df[col].loc[df[[col]].duplicated(keep=False) == 1]
        dup_unique = dup.nunique()
        dup_full = len(dup)
        if dup_unique > 0:
            print("{}: {} unique duplicate values ({} total duplicates)" \
                .format(df[col].name, dup_unique, dup_full))




### OUTLIERS - Count and Removal


def count_outliers_IQR_method(df, outlier_cols=None, IQR_dist = 1.5):
    """Display outlier count in specified columns depending on distance 
    from 1th / 3rd quartile. NaN are ignored.

    ARGUMENTS:
        df: DataFrame
        outlier_cols: List with columns to clean (default: all num columns)
        IQR_dist: Float for cut-off distance from quartiles (default: 1.5 * IQR)
    """
    outlier_cols = outlier_cols if outlier_cols is not None else \
        list(df.select_dtypes(include = ['float64', 'int64']).columns)
    for col in outlier_cols:
        q25, q75 = np.nanpercentile(df[col], 25), np.nanpercentile(df[col], 75)
        iqr = q75 - q25
        # calculate the outlier cut-off
        cut_off = iqr * IQR_dist
        lower, upper = q25 - cut_off, q75 + cut_off
        # identify outliers
        outliers = [x for x in df[col] if x < lower or x > upper]
        print(col+'\nIdentified outliers: {}'.format(len(outliers)))
        print('Percentage of outliers: {:.1f}%\n'.format(
            (len(outliers)/len(df[col]))*100))


def remove_outliers_IQR_method(df, outlier_cols=None , IQR_dist = 1.5):
    """Remove outliers in specified columns depending on distance from 
    1th / 3rd quartile. NaN are ignored. Returns transformed DataFrame.
    
    ARGUMENTS:
        df: DataFrame
        outlier_cols: List with columns to clean (default: all num columns)
        IQR_dist: Float for cut-off distance from quartiles (default: 1.5 * IQR)
    """
    outlier_cols = outlier_cols if outlier_cols is not None else \
            list(df.select_dtypes(include = ['float64', 'int64']).columns)
    outer_row_count_1 = len(df)
    for col in outlier_cols:
        print(col)
        row_count_1 = len(df)
        distance = IQR_dist * (np.nanpercentile(df[col], 75) - np.nanpercentile(df[col], 25)) 
        df.drop(df[df[col] > distance + np.nanpercentile(df[col], 75)].index, inplace=True)
        df.drop(df[df[col] < np.nanpercentile(df[col], 25) - distance].index, inplace=True)
        row_count_2 = len(df)
        print("Rows removed: {}\n".format(row_count_1 - row_count_2))
    outer_row_count_2 = len(df)
    print("\nRows removed in total: {}\n" \
        .format(outer_row_count_1 - outer_row_count_2))


### TRANSFORMATION

def apply_log10 (df, cols_to_log10=None, treat_NaN=False):
    """Transform values of selected columns to Log10. NaN are not 
    affected by default, parameter can be changed. Returns transformed 
    DataFrame, column names have "_log" appended.
    Params
    ======
        df: DataFrame
        cols_to_log10: list of columns that will have log10 
            transformation applied. Default is all numerical columns.
        treat_NaN: sets NaN to small negative value, default is False.
    """
    cols_to_log10 = cols_to_log10 if cols_to_log10 is not None else \
               list(df.select_dtypes(include = ['float64', 'int64']).columns)
    for col in df[cols_to_log10]:
        if col in df:
            df[col] = df[col].apply(lambda x: np.log10(max(x,1)))
            if treat_NaN:
                df[col].replace(np.nan, -1, inplace=True)
        else:
            display(col + " not found")
        #rename log-transformed columns
        df.rename(columns={col: col+'_log'}, inplace=True)
