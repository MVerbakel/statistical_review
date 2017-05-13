
import pandas as pd
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from scipy.stats import sem
import scipy.stats as scs
import seaborn as sns
import brewer2mpl
plt.style.use('ggplot')

color_options = brewer2mpl.get_map('Paired', 'qualitative', 8).mpl_colors


# Print overall stats about the data frame

def df_stats(df):
    """
    Calculates and prints key statistics for a DataFrame
    :param df: A Pandas DataFrame
    :return: number of rows, number of columns
    """
    rows_n = df.shape[0]
    columns_n = df.shape[1]

    print()
    print('The data set contains {} rows and {} columns'.format(rows_n, columns_n))
    print()
    print('Summary:')
    print(df.info())
    print()

    return rows_n, columns_n


# Correct data types

# TODO: Handle date formats
# latest_data['date'] = pd.to_datetime(latest_data['date'], format = "%Y-%m-%d")

def correct_dtypes(df, dtype_dictionary):
    """
    Takes a dictionary containing feature and data type pairs, and converts the data type
    in the given DataFrame accordingly
    :param df: DataFrame containing columns you wish to change the dtype of
    :param dtype_dictionary: A dictionary of column_name, dtype pairs (e.g. age: np.int64)
    :return: Returns the original DataFrame with the dtypes corrected
    """
    for key, value in dtype_dictionary.items():
        df[key] = df[key].astype(value)
    return df


# Provide stats on the target

def analyse_target(target):
    """
    Return the count and percentage of cases in each target value
    :param target: column that contains the variable of interest
    :return: DataFrame with the count and percent
    """
    counts = pd.DataFrame(target.value_counts())
    full_counts = len(target)
    counts['percent'] = counts/full_counts
    counts.columns = ['count', 'percent']
    counts.sort_values(by=['percent'], ascending=[0])
    print(counts)
    print()
    return counts


# Basic data summary

def describe_data(df):

    d = df.describe(include='all')

    return d


# Review NA

def count_na(df):
    """
    Count null values in columns of a DataFrame
    :param df: Pandas DataFrame
    :return: Return both raw count and percentage of null values per column
    """
    na_counts = pd.DataFrame(df.isnull().sum(), columns=['Count'])
    full_counts = len(df)
    percent = na_counts.Count / full_counts
    na_counts['Percent'] = percent
    return na_counts

# TODO: Provide a more advanced na imputation using prediction e.g. mice - for training only


# Clean up categorical features for analysis purposes

def clean_cats(df):
    """
    Takes categorical columns and groups values with <10% of cases into an 'other' category
    :param df: Pandas DataFrame for cleaning
    :return: Pandas DataFrame with cleaned categorical columns
    """
    categorical_cols = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['object', 'str']]
    threshold = len(df) * 0.1
    new_df = df.apply(lambda x: x.mask(x.map(x.value_counts()) < threshold, 'other')
                      if x.name in categorical_cols else x)

    return new_df


# TODO: Handle date/time features
# line chart/ bars by month for date

def plot_features(df, target_col_str):
    """
    Creates a set of plots for analysing numerical and categorical features compared to a categorical target
    :param df: Pandas DataFrame
    :param target_col_str: name of the target feature (string data type)
    :return: Matrix of plots for features in df
    """
    # TODO: Handle numerical targets
    features = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['float64', 'int64', 'object', 'str']]
    target_labels = df[target_col_str].unique()

    rows_n = len(features)
    columns_n = 3
    fig, axs = plt.subplots(rows_n, columns_n, figsize=(columns_n*4, rows_n*2), sharey=False, sharex=False)

    def create_plot(col_n, row_n, col_name):

        if col_n is 0 and df[col_name].dtype in ['object', 'str']:
            print(col_name)
            value_counts = 100.0 * (df[col_name].value_counts(dropna=False) / len(df[col_name]))
            unique_val = df[col_name].unique()
            ind = np.arange(len(unique_val))
            axs[row_n][col_n].bar(ind, value_counts, align='center', color='deepskyblue')
            axs[row_n][col_n].set_title(col_name)
            axs[row_n][col_n].set_xticks(ind)
            axs[row_n][col_n].set_xticklabels(unique_val)
            axs[row_n][col_n].set_ylabel('% of total')

        elif col_n is 0 and df[col_name].dtype in ['float64', 'int64']:
            axs[row_n][col_n].hist(df[col_name], color='deepskyblue')
            axs[row_n][col_n].set_title(col_name)
            axs[row_n][col_n].set_ylabel('Count')

        elif col_n is 1 and df[col_name].dtype in ['object', 'str']:

            width = 0.3
            i = 1
            c = 0

            for l in target_labels:
                pos = width * i
                data = df.where(df[target_col_str] == l)
                unique_val = data[col_name].unique()
                ind = np.arange(len(unique_val)) + pos
                value_counts = 100.0 * (data[col_name].value_counts(dropna=False) / len(data[col_name]))
                axs[row_n][col_n].bar(left=ind, height=value_counts, width=width,
                                      color=color_options[c],  label=l)
                axs[row_n][col_n].set_title(col_name)
                axs[row_n][col_n].set_xticks(ind)
                axs[row_n][col_n].set_xticklabels(unique_val)
                axs[row_n][col_n].set_ylabel('% of total')
                axs[row_n][col_n].legend()
                i += 1
                c += 1

        elif col_n is 1 and df[col_name].dtype in ['float64', 'int64']:

            data_by_target = []

            for l in target_labels:
                data = df.where(df[target_col_str] == l)
                data = data[~np.isnan(data[col_name])][col_name].as_matrix()
                data_by_target.append(data)

            axs[row_n][col_n].hist(data_by_target, label=target_labels,
                                   color=color_options[0:len(target_labels)], alpha=0.7)
            axs[row_n][col_n].set_title(col_name)
            axs[row_n][col_n].set_ylabel('Count')
            axs[row_n][col_n].legend()

        elif col_n is 2 and df[col_name].dtype in ['object', 'str']:

            width = 0.3
            i = 1
            c = 0

            for l in target_labels:
                pos = width * i
                data = df.where(df[target_col_str] == l)
                unique_val = data[col_name].unique()
                ind = np.arange(len(unique_val)) + pos
                value_counts = data[col_name].value_counts(dropna=False)
                axs[row_n][col_n].bar(left=ind, height=value_counts, width=width,
                                      color=color_options[c],  label=l)
                axs[row_n][col_n].set_title(col_name)
                axs[row_n][col_n].set_xticks(ind)
                axs[row_n][col_n].set_xticklabels(unique_val)
                axs[row_n][col_n].set_ylabel('Count')
                axs[row_n][col_n].legend()
                i += 1
                c += 1

        elif col_n is 2 and df[col_name].dtype in ['float64', 'int64']:

            data_by_target = []

            for l in target_labels:
                data = df.where(df[target_col_str] == l)
                data = data[~np.isnan(data[col_name])][col_name].as_matrix()
                data_by_target.append(data)

            fp = dict(marker='o', markersize=5, linestyle='none')
            mp = dict(marker='D', markersize=5, markerfacecolor='blue')
            bp = axs[row_n][col_n].boxplot(data_by_target, labels=target_labels, sym='k.', showfliers=True,
                                           flierprops=fp, showmeans=True, meanline=False, meanprops=mp)

            for b in bp['boxes']:
                b.set(color='deepskyblue', linewidth=2)
            for w in bp['whiskers']:
                w.set(color='deepskyblue', linewidth=2)
            for cap in bp['caps']:
                cap.set(color='deepskyblue', linewidth=2)
            for median in bp['medians']:
                median.set(color='red', linewidth=2)

            axs[row_n][col_n].set_title(col_name)
            axs[row_n][col_n].set_ylabel(col_name)
            l1 = mlines.Line2D([], [], color='blue', marker='D', markersize=5, label='Mean', linestyle='None')
            l2 = mlines.Line2D([], [], color='red', marker='None', label='Median')
            l3 = mlines.Line2D([], [], color='black', marker='o', markersize=5, label='Outliers', linestyle='None')
            axs[row_n][col_n].legend(handles=[l1, l2, l3], loc=9, fontsize=8, ncol=3, frameon=True)

        else:
            pass
            # axs[row_n][col_n].plot(df[col_name])
            # axs[row_n][col_n].set_title(col_name)

        return axs

    for i in range(rows_n):

        f = features[i]

        for j in range(columns_n):
            create_plot(j, i, f)


# Statistical tests for relationship with target

# Chi-square for cat vs cat

def chi_squared(df, target_col_str):
    """
    Returns a cross-tab and chi-squared test for each feature with dtype 'object' or 'str'
    :param df: Pandas DataFrame with target and categorical columns
    :param target_col_str: name of the target column
    :return: p values for chi-squared test
    """

    print('Ho: No significant relationship, feature is independent of target')
    print()
    categorical_cols = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['object', 'str']]
    chi_dict = {}

    for c in categorical_cols:

        ct = pd.crosstab(df[target_col_str], df[c])
        ct_dec = ct.apply(lambda r: r/r.sum(), axis=0)*100
        ct_dec_f = ct_dec.applymap(lambda x: '%.0f' % x)
        chi2, p, dof, expected = scs.chi2_contingency(ct)
        chi_dict[c] = p
        exp_under_five_percent = np.sum(expected < 5) / len(expected.flat)

        if exp_under_five_percent > 0.2:
            print('Warning: more than 20% of expected values less than 5, results invalid')
        else:
            pass

        print(c)
        print(' ')
        print('Outcome by' + ' ' + str(c) + ' ' + '(n)')
        print(ct)
        print(' ')
        print('Outcome by' + ' ' + str(c) + ' ' + '(%)')
        print(ct_dec_f)
        print(' ')
        print(c, 'Chi-squared test p value: %.4f' % p)

        if p < 0.05:
            print('Significant (95% CI)! Reject null hypothesis, evidence suggests target is dependent on this feature')
        elif p >= 0.05:
            print('Insufficient evidence to reject null hypothesis (95% CI). Suggests target and feature independent')
        else:
            print('Error in assessment of p-value')

        print(' ')

    chi_df = pd.DataFrame.from_dict(chi_dict, orient='index')
    chi_df.columns = ['p_value']

    plt.style.use('fivethirtyeight')
    chi_df['p_value'].sort_values(ascending=True).plot(kind='barh', use_index=True, color='deepskyblue',
                                                       title='Chi-squared Test for Independence with Target')

    plt.xlabel('p-value')
    plt.xscale('log')
    p_line = plt.axvline(x=0.05, color='darkorange', label='p-value cut-off', lw=2)
    plt.legend([p_line], ['Significant p-value\ncut-off (95% CI)'], bbox_to_anchor=(1.05, 1), loc=2)
    plt.grid(False)
    plt.show()

    return chi_df


# num vs num

def correlation_plot(df, ax):
    """
    Plots the correlation between numerical features in data
    :param df: Pandas DataFrame
    :param ax: Axes object for figure
    :return: correlations
    """

    numerical_cols = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['float64', 'int64']]
    corr = df[numerical_cols].corr()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, cmap=cmap, square=True, annot=True, annot_kws={"size": 10}, linewidths=.5,
                cbar_kws={"shrink": .5}, ax=ax, fmt='.2f')

    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.title('Correlation plot of numerical features')

    return corr


# Numerical features vs categorical target

# TODO: 2 sample t-test, or ANOVA depending on n of target categories

def paired_plot(df, target_col_str):
    """
    Plots the relationship between numerical features in data by the target
    :param df: Pandas DataFrame - recommend limiting the number of features to a max of 10 at a time
    :param target_col_str: Name of the target feature (string format)
    :return: paired plot grid
    """
    numerical_cols = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['float64', 'int64']]
    # numerical_cols.append(target_col_str) # not needed if converted to numerical dtype

    data = df[numerical_cols].copy()
    pp_grid = sns.pairplot(data, hue=target_col_str, size=1)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.title('Paired plot of numerical features and target')

    return pp_grid
