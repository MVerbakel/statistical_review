
import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Walk-through of standard analysis before modelling

# Set up formatting
pd.set_option('display.float_format', lambda x: '%.3f' % x)
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

# Drop features that are not meaningful for the analysis
drop_elements = ['ID', 'Unnamed: 0']
example_df = example_df.drop(drop_elements, axis=1)

# Review target
target = example_df['Y_2']
print('\nTarget Summary:')
stats.analyse_target(target=target)

# TODO: make sure target is numerical - encode

# Review key stats for features
print('\nFeature Summary:')
print(example_df.describe(include='all'))

# Review NA values
data_na = stats.count_na(example_df)
print(data_na)

high_na = data_na.loc[data_na['Percent'] > 0.9].index.tolist()
print('Features with > 90% missing values to be dropped: {}'.format(len(high_na)))
print(high_na)
example_df = example_df.drop(high_na, axis=1)

# Basic NA cleaning

categorical_cols = example_df.select_dtypes(['object']).columns
numerical_cols = example_df.select_dtypes(include=[np.number]).columns

example_df.fillna({x: 'unknwn' for x in categorical_cols}, inplace=True)
example_df.fillna({x: -1 for x in numerical_cols}, inplace=True)

# Group infrequent values in categorical features to 'other'
example_df = stats.clean_cats(example_df)

# Check for relationship between target and other features

# Numerical vs Numerical Features - Correlation

high_corr_df = stats.correlated_features(df=example_df, min_corr=0.85)
high_corr_unique_f = list(set(high_corr_df['feature 1'].unique().tolist()
                              + high_corr_df['feature 2'].unique().tolist()))

print('Number of highly correlated features: {}'.format(len(high_corr_unique_f)))
print(high_corr_unique_f)

# Quick solution to reduce correlated features - drop the 2nd feature

high_corr_f2 = high_corr_df['feature 2'].unique().tolist()
example_df = example_df.drop(high_corr_f2, axis=1)

# Plot correlations between remaining features

f, ax = plt.subplots(figsize=(10, 10))
correlations = stats.correlation_plot(example_df, ax=ax)
plt.show()

# Remaining Numerical Features vs Categorical Target

pbs_corr = stats.point_biserial_corr(df=example_df, target_col_str='Y_2')
title = 'Numerical feature correlation with target (top 20)'
score_label = 'correlation coefficient (r)'

stats.plot_score(df=pbs_corr, score_col='corr', n=20, title=title, log_scale=False, score_label=score_label)

# Visual analysis of Top Numerical Features vs Target

top_numerical_f = pbs_corr.sort_values(by='corr_abs', ascending=False)[0:20].index.tolist()
top_numerical_f.append('Y_2')
numerical_df_p = example_df[top_numerical_f]
stats.plot_features(df=numerical_df_p, target_col_str='Y_2')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

# Categorical vs Categorical - chi squared test of independence

print('\nChi-squared Test for Independence:')
chi_square_test = stats.chi_squared(example_df, target_col_str='Y_2')

failed_chi = chi_square_test.loc[chi_square_test['p_value'] >= 0.05]
failed_chi_f = failed_chi.index.tolist()
print('Accept null hypothesis - no apparent association with target:')
print(failed_chi_f)

passed_chi = chi_square_test.loc[chi_square_test['p_value'] < 0.05]
passed_chi_f = passed_chi.index.tolist()
print('Reject null hypothesis - apparent association with target:')
print(passed_chi_f)

invalid_chi = chi_square_test.loc[chi_square_test['p_value'].isnull()]
invalid_chi_f = invalid_chi.index.tolist()
print('Did not meet assumptions for test (>20% of expected counts <5):')
print(invalid_chi_f)
# need to divert these to another test (e.g. Fishers Exact)

# Visual analysis of Categorical Features

top_categorical_f = passed_chi_f + invalid_chi_f + 'Y_2'

categorical_df_p = example_df[top_categorical_f]
stats.plot_features(df=categorical_df_p, target_col_str='Y_2')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()
