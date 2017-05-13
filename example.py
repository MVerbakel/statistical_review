
import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Walk-through of standard analysis before modelling

# Set up formatting
pd.set_option('display.float_format', lambda x: '%.2f' % x)
plt.style.use('ggplot')

# Load data
example_df = pd.read_csv('bank.csv', na_values=[np.nan])

# Print high-level summary of the data frame
print('Data Summary:')
rows_n, columns_n = stats.df_stats(example_df)

# Preview data
print('Preview of Data:')
print(example_df.head(3))
print(example_df.tail(3))
print()

# After reviewing data types, correct any that are wrong by putting them in a dictionary with the correct type
dtype_dictionary = {'ID':           np.object,
                    'Age_2':        np.int64,
                    'Balance_2':    np.float,
                    'Day_2':        np.int64,
                    'Marital_2':    np.object,
                    'Education_2':  np.object}

example_df = stats.correct_dtypes(df=example_df, dtype_dictionary=dtype_dictionary)
print('Corrected Data Types:\n{}'.format(example_df.dtypes))
print()

# Drop features that are not meaningful for the analysis
drop_elements = ['ID', 'Unnamed: 0']
example_df = example_df.drop(drop_elements, axis=1)

# Review target
target = example_df['Y_2']
print('Target Summary:')
print()
stats.analyse_target(target=target)

# TODO: make sure target is numerical - encode

# Review key stats for features
print('Feature Summary:')
print()
print(stats.describe_data(example_df))
print()

# Review NA values
data_na = stats.count_na(example_df)
print(data_na)

# Basic NA cleaning

for col in example_df:
    dt = example_df[col].dtype
    if dt == int or dt == float:
        example_df[col].fillna(0)
    else:
        example_df[col].fillna('unknown')

# Group infrequent values in categorical features to 'other'
example_df = stats.clean_cats(example_df)

# Check for relationship between target and other features
# Numerical vs Numerical - correlation
f, ax = plt.subplots(figsize=(10, 10))
correlations = stats.correlation_plot(example_df, ax=ax)
plt.show()

# Numerical vs Categorical - paired plot
paired_grid = stats.paired_plot(example_df, target_col_str='Y_2')
plt.show()

# Categorical vs Categorical - chi squared test of independence
print()
print('Chi-squared Test for Independence:')
chi_square_test = stats.chi_squared(example_df, target_col_str='Y_2')

# Visual analysis
reduced_df = example_df[['Y_2', 'Age_2', 'Balance_2', 'Marital_2', 'Education_2']].copy()
stats.plot_features(reduced_df, 'Y_2')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()


