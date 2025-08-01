
# Function to summaries data frames in terms of counts, unique values, non-null values, null values, and data types after initial data exploration and cleaning
def get_df_summary(df):
    """
    Generates a summary of the DataFrame with counts, unique values, non-null values, null values, and data types for each column.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to summarize.
    
    Returns:
    pd.DataFrame: A DataFrame containing the summary statistics for each column.
    """
    # import necessary packages
    import pandas as pd
    
    # Initialize the output DataFrame with appropriate columns
    output_df = pd.DataFrame(columns=['Feature',
                                      'Number of rows',
                                      'Non-null values',
                                      'Null values',
                                      'Null values (%)',
                                      'Number of unique values',                                      
                                      'Data type'])
    
    output_df.set_index('Feature', inplace=True)
    
    # Duplicated rows in the data frame
    n_duplicated = df.duplicated().sum()
    indexes_duplicated = df[df.duplicated()].index.to_list()
    print(f'Number of duplicate entries, except the first one: {n_duplicated}')
    [print(f'Indexes of duplicate entries, except the first one: {indexes_duplicated}') if n_duplicated > 0 else print(f'Indexes of duplicated entries, except the first one: None')]
    
    # Iterate over each column in the input DataFrame
    for col in df:
        rows = df.shape[0]
        count =df[col].count() # Count Non-NA cells, same output as df[col].notnull().sum()
        unique_val = df[col].nunique()
        not_null_val = df[col].notnull().sum() # same output as df[col].count()
        null_val = df[col].isnull().sum() # same output as df[col].isna().sum()
        null_val_per = (null_val/(null_val+not_null_val))*100
        data_type = df[col].dtype
        
        
        output_df.loc[col] = [rows,
                              count,
                              null_val, 
                              round(null_val_per, 4), 
                              unique_val, 
                              data_type] 
        
    return output_df.sort_values(by='Null values')


# A function to perform initial univariate exploratory data analysis on a pandas DataFrame.
# It calculates summary statistics for each feature, including count, missing values, unique values, mode, and various statistics for numerical features such as min, max, mean, std, skewness, and kurtosis.
# It also generates visualizations such as histograms and box plots for numerical features, and count plots for categorical features.
# This function is useful for determining which features may need to be dropped based on their distribution.
def get_df_univariate_stats(df, n_bins='auto', with_figures=False):
    """
    Perform initial univariate exploratory data analysis on a pandas DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    pd.DataFrame: A DataFrame containing summary statistics and plots for each feature.
    
    Useful:
        - To determine which features are to be dropped based on their distribution 
    
    cautious:
    Make sure that the pandas DataFrame has gone through initial data cleaning process, such 
    as standardizing column names, column data types, 
    prior to using the function.
    """
    # Importing necessary packages
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Define the columns for the output DataFrame
    output_df = pd.DataFrame(columns=['feature',                                # column names as features
                                      'type', 'count','missing', 'unique','mode',     # For both numerical and categorical data types
                                      'min','q1','median','q3','max','mean','std','skewness','kurtosis'] # Only for numerical data types
                             )
    output_df.set_index('feature', inplace=True)
    
    # Define font sizes for text in figures
    title_fontsize = 16
    tick_label_fontsize = 12
    
       
    for col in df:
                
        
        # Calculate metics that apply to all dtypes
        data_type = df[col].dtype
        count = df[col].count()
        missing = df[col].isna().sum()
        unique  = df[col].nunique()
        #mode =  df[col].mode()
        mode = df[col].mode().iloc[0] if not df[col].mode().empty else None
         
        if pd.isna(df[col]).all():
            print(f'"{col}" is empty')
            continue
        
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)
            
        # Initialize metric for numeric columns
        min_val, q1, median, q3, max_val, mean, std, skewness, kurt = [None]*9          
        
        if pd.api.types.is_numeric_dtype(df[col]):
            
            # Calculate metrics that apply only to numeric features
            
            min_val = df[col].min()
            q1 = df[col].quantile(0.25)
            median = df[col].median()
            q3 = df[col].quantile(0.75)
            max_val = df[col].max()
            mean = df[col].mean()
            std = df[col].std()
            skewness = df[col].skew()
            kurt = df[col].kurt()
            
            # Plot histogram for numeric features

            #plt.figure(figsize=(10, 4))
            ##plt.rcParams["figure.autolayout"] = True
            #sns.histplot(df[col], kde=True)
            #plt.xticks(rotation=45, ha='right')
            #plt.title(f'Distribution of {col}')
            #plt.tight_layout()
            #plt.show()
            if with_figures==True:               
                
                fig, axes = plt.subplots(2,1,
                            figsize=(10,6),
                            gridspec_kw={'height_ratios': [3, 1]},
                            )
                # Plot histogram with KDE
                sns.histplot(x=df[col],
                bins=n_bins, #df[col].unique().size,
                kde=True,
                ax=axes[0],
                )
                axes[0].set_title(f'Distribution of {col}',
                                fontsize=title_fontsize)
                axes[0].tick_params(axis='x',
                                    rotation=45,
                                    labelsize=tick_label_fontsize,)
                axes[0].tick_params(axis='y',
                                    labelsize=tick_label_fontsize)        
                axes[0].set_ylabel('Count',
                                fontsize=tick_label_fontsize)
                axes[0].set_xlabel(col,
                                fontsize=tick_label_fontsize)
            
                # Get the counts and adjust y-axis limits
                #counts, _ = np.histogram(df[col].dropna())
                #max_count = counts.max()
                
                # Set the y-axis limit to an appropriate value
                #max_count = df[col].value_counts().max() <- This is wrong as upper limit will depend on the unique value counts within the size of a bin
                                                        #<- A bin doesn't represent a unique value, it may include several unique values so each their counts can go beyond df[col].value_counts().max()  
                #axes[0].set_ylim(0, max_count + max_count * 0.5)     
                            
                # Get the x-axis limits from the histogram
                x_limits = axes[0].get_xlim()       # To use with (1)
            
                # Plot box plot
                sns.boxplot(df[col],
                            orient='h',
                            ax=axes[1],)
                
                # lower_bound = q1 - 1.5 * (q3 - q1)
                # upper_bound = q3 + 1.5 * (q3 - q1)
                
                # plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound', axes=axes[1])
                # plt.axvline(upper_bound, color='green', linestyle='--', label='Upper Bound', axes=axes[1])
                
                x_ticks = [df[col].min(),
                        df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)), # lower bound of outliers
                        df[col].quantile(0.25),
                        df[col].quantile(0.5),
                        df[col].quantile(0.75),
                        df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)), # upper bound of outliers
                        df[col].max()]                
                
                axes[1].set_title(f'Distribution (box plot) of {col}',
                                fontsize=title_fontsize)
                axes[1].set_xticks(x_ticks)
                #axes[1].set_xticklabels(['lb_outliers', 'Min', 'Q1', 'Median', 'Q3', 'ub_outliers', 'Max'], rotation=45, ha='right')
                axes[1].tick_params(axis='x', 
                                    rotation=45,
                                    labelsize=tick_label_fontsize,)
                axes[1].tick_params(axis='y',
                                    labelsize=tick_label_fontsize,)
        
                axes[1].set_xlabel(col,
                                fontsize=tick_label_fontsize)
                
                # Set the same x-axis limits for the box plot
                axes[1].set_xlim(x_limits)         # <--(1)
                
                # Get common x-axis range
                #x_min = df[col].min()
                #x_max = df[col].max()
        
                # Set common x-axis range for both subplots
                #axes[0].set_xlim(x_min, x_max)
                #axes[1].set_xlim(x_min, x_max)  
                        
                plt.tight_layout()
                plt.show()
#            else:
#                output_df.loc[col] = [data_type, count, missing, unique, mode,
#                                      min_val, q1, median, q3, max_val, mean, std, skewness, kurt,]       
            
        else:
            if with_figures==True:
                
                # Plot countplot for categorical features
                
                fig, axes = plt.subplots(figsize=(10,6),)

#                plt.figure(figsize=(20, 8))
                                
                sns.countplot(x=df[col], order = df[col] \
                    .value_counts().index)
                
#                axes.set_xticks(x_ticks)
                #axes.set_xticklabels(['Min', 'Q1', 'Median', 'Q3', 'Outlier', 'Max'], rotation=45, ha='right')
                
                axes.tick_params(axis='x', 
                                    rotation=45,
                                    labelsize=tick_label_fontsize,)
                
                axes.tick_params(axis='y',
                                    labelsize=tick_label_fontsize,)
        
                axes.set_xlabel(col,
                                fontsize=tick_label_fontsize)
                
                axes.set_ylabel('Count',
                                fontsize=tick_label_fontsize)
                
                axes.set_title(f'Distribution of {col}',
                               fontsize=title_fontsize)
                
                
                plt.tight_layout()
                plt.show()
#            else:
#                output_df.loc[col] = [data_type, count, missing, unique, mode,
#                                      min_val, q1, median, q3, max_val, mean, std, skewness, kurt,]

        
        # Append the calculated metrics to the output DataFrame
        output_df.loc[col] = [
            data_type, count, missing, unique, mode,
            min_val, q1, median, q3, max_val, mean, std, skewness, kurt,
        ]
        
        

    return output_df 
