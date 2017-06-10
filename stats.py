
import pandas as pd
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import scipy.stats as scs
import seaborn as sns
import brewer2mpl
from scipy.stats import pointbiserialr

plt.style.use('ggplot')

color_options = brewer2mpl.get_map('Paired', 'qualitative', 8).mpl_colors

# TODO: more statistical tests and assumptions


def df_stats(df):
    """
    Calculates and prints key statistics for a DataFrame
    :param df: A Pandas DataFrame
    :return: number of rows, number of columns
    """
    rows_n = df.shape[0]
    columns_n = df.shape[1]

    print('The data set contains {} rows and {} columns\n'.format(rows_n, columns_n))
    print('Summary:')
    print(df.info())

    return rows_n, columns_n


def correct_dtypes(df, dtype_dictionary):
    """
    Corrects data types for features/variables (columns) in a DataFrame. Takes a dictionary containing
    feature and data type pairs, and converts the data type accordingly
    :param df: DataFrame containing columns you wish to correct
    :param dtype_dictionary: A dictionary of column_name, dtype pairs (e.g. age: np.int64)
    :return: Returns the original DataFrame with the dtypes corrected
    """
    # TODO: Handle date formats
    # latest_data['date'] = pd.to_datetime(latest_data['date'], format = "%Y-%m-%d")
    for key, value in dtype_dictionary.items():
        df[key] = df[key].astype(value)
    return df


def analyse_target(target):
    """
    Return the count and percentage of cases (rows) in each target category
    :param target: column that contains the variable of interest
    :return: DataFrame with the count and percent
    """
    # TODO: Add alternate calculations for numerical targets
    counts = pd.DataFrame(target.value_counts())
    full_counts = len(target)
    counts['percent'] = counts/full_counts
    counts.columns = ['count', 'percent']
    counts.sort_values(by=['percent'], ascending=[0])
    print(counts)
    return counts


def count_na(df):
    """
    Count missing (null) values in columns of a DataFrame
    :param df: Pandas DataFrame with missing values read in as nan e.g. pd.read_csv('data.csv', na_values=[np.nan])
    :return: Return both raw count and percentage of null values per column
    """
    # TODO: Provide a more advanced na imputation using prediction e.g. mice - for training only
    na_counts = pd.DataFrame(df.isnull().sum(), columns=['Count'])
    full_counts = len(df)
    percent = na_counts.Count / full_counts
    na_counts['Percent'] = percent
    na_counts.sort_values(by='Percent', ascending=False)

    return na_counts


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


def single_bar_percent(df, col_name, axs, col_n, row_n):
    """
    Creates a bar chart for a Pandas data series, y axis is %
    :param df: Pandas DataFrame
    :param col_name: Name of the column to be plotted
    :param axs: The axes object for plotting
    :param col_n: Column position in axs object
    :param row_n: Row position in axs object
    :return: axes object
    """
    value_counts = 100.0 * (df[col_name].value_counts(dropna=False) / len(df[col_name]))
    unique_val = df[col_name].unique()
    ind = np.arange(len(unique_val))
    axs[row_n][col_n].bar(ind, value_counts, align='center', color='deepskyblue')
    axs[row_n][col_n].set_title(col_name)
    axs[row_n][col_n].set_xticks(ind)
    axs[row_n][col_n].set_xticklabels(unique_val)
    axs[row_n][col_n].set_ylabel('% of total')

    return axs


def single_hist(df, col_name, axs, col_n, row_n):
    """
    Creates a histogram for a Pandas data series
    :param df: Pandas DataFrame
    :param col_name: Name of the column to be plotted
    :param axs: The axes object for plotting
    :param col_n: Column position in axs object
    :param row_n: Row position in axs object
    :return: axes object
    """
    axs[row_n][col_n].hist(df[col_name], color='deepskyblue')
    axs[row_n][col_n].set_title(col_name)
    axs[row_n][col_n].set_ylabel('Count')

    return axs


def multi_bar_percent(df, col_name, axs, col_n, row_n, target_col_str, target_labels):
    """
    Creates a bar chart for a Pandas data series split by another series (target), y axis is %
    :param df: Pandas DataFrame
    :param col_name: Name of the column to be plotted
    :param axs: The axes object for plotting
    :param col_n: Column position in axs object
    :param row_n: Row position in axs object
    :param target_col_str: string name of the column containing the target
    :param target_labels: list of unique target values (strings)
    :return: axes object
    """

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
                              color=color_options[c], label=l)
        axs[row_n][col_n].set_title(col_name)
        axs[row_n][col_n].set_xticks(ind)
        axs[row_n][col_n].set_xticklabels(unique_val, rotation=90)
        axs[row_n][col_n].set_ylabel('% of total')
        axs[row_n][col_n].legend()
        i += 1
        c += 1

    return axs


def multi_hist(df, col_name, axs, col_n, row_n, target_col_str, target_labels):
    """
    Creates a histogram for a Pandas data series split by another series (target)
    :param df: Pandas DataFrame
    :param col_name: Name of the column to be plotted
    :param axs: The axes object for plotting
    :param col_n: Column position in axs object
    :param row_n: Row position in axs object
    :param target_col_str: string name of the column containing the target
    :param target_labels: list of unique target values (strings)
    :return: axes object
    """
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

    return axs


def multi_bar_count(df, col_name, axs, col_n, row_n, target_col_str, target_labels):
    """
    Creates a bar chart for a Pandas data series split by another series (target), y axis is counts
    :param df: Pandas DataFrame
    :param col_name: Name of the column to be plotted
    :param axs: The axes object for plotting
    :param col_n: Column position in axs object
    :param row_n: Row position in axs object
    :param target_col_str: string name of the column containing the target
    :param target_labels: list of unique target values (strings)
    :return: axes object
    """
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
                              color=color_options[c], label=l)
        axs[row_n][col_n].set_title(col_name)
        axs[row_n][col_n].set_xticks(ind)
        axs[row_n][col_n].set_xticklabels(unique_val, rotation=90)
        axs[row_n][col_n].set_ylabel('Count')
        axs[row_n][col_n].legend()
        i += 1
        c += 1

    return axs


def multi_boxplot(df, col_name, axs, col_n, row_n, target_col_str, target_labels):
    """
    Creates a boxplot for a Pandas data series split by another series (target)
    :param df: Pandas DataFrame
    :param col_name: Name of the column to be plotted
    :param axs: The axes object for plotting
    :param col_n: Column position in axs object
    :param row_n: Row position in axs object
    :param target_col_str: string name of the column containing the target
    :param target_labels: list of unique target values (strings)
    :return: axes object
    """

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

    return axs


def create_plot(df, col_name, axs, col_n, row_n, target_col_str, target_labels):
    """
    Creates plots to fit into a set of subplots based on the data type and position
    :param df: Pandas DataFrame
    :param col_name: Name of the column to be plotted
    :param axs: The axes object for plotting
    :param col_n: Column position in axs object
    :param row_n: Row position in axs object
    :param target_col_str: string name of the column containing the target
    :param target_labels: list of unique target values (strings)
    :return: axes object
    """

    if col_n is 0 and df[col_name].dtype in ['object', 'str']:
        single_bar_percent(df, col_name, axs, col_n, row_n)

    elif col_n is 0 and df[col_name].dtype in ['float64', 'int64']:
        single_hist(df, col_name, axs, col_n, row_n)

    elif col_n is 1 and df[col_name].dtype in ['object', 'str']:
        multi_bar_percent(df, col_name, axs, col_n, row_n, target_col_str, target_labels)

    elif col_n is 1 and df[col_name].dtype in ['float64', 'int64']:
        multi_hist(df, col_name, axs, col_n, row_n, target_col_str, target_labels)

    elif col_n is 2 and df[col_name].dtype in ['object', 'str']:
        multi_bar_count(df, col_name, axs, col_n, row_n, target_col_str, target_labels)

    elif col_n is 2 and df[col_name].dtype in ['float64', 'int64']:
        multi_boxplot(df, col_name, axs, col_n, row_n, target_col_str, target_labels)

    else:
        pass

    return axs


def plot_features(df, target_col_str):
    """
    Creates a set of plots for analysing numerical and categorical features compared to a categorical target
    :param df: Pandas DataFrame
    :param target_col_str: name of the target feature (string data type)
    :return: Matrix of plots for features in df
    """
    # TODO: Handle numerical targets
    # TODO: Handle date/time features - line chart/ bars by month for date
    features = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['float64', 'int64', 'object', 'str']]
    target_labels = df[target_col_str].unique()

    rows_n = len(features)
    columns_n = 3
    fig, axs = plt.subplots(rows_n, columns_n, figsize=(columns_n*4, rows_n*2), sharey=False, sharex=False)

    for i in range(rows_n):

        f = features[i]

        for j in range(columns_n):
            create_plot(df=df, col_name=f, axs=axs, col_n=j, row_n=i,
                        target_col_str=target_col_str, target_labels=target_labels)


def chi_squared(df, target_col_str):
    """
    Returns a cross-tab and chi-squared test for each feature with dtype 'object' or 'str' and plots results
    Use to test for a relationship between each categorical feature and a categorical target
    :param df: Pandas DataFrame with target and categorical columns
    :param target_col_str: name of the target column
    :return: p values for chi-squared test
    """

    print('Ho: No significant relationship, feature is independent of target')
    print()

    # categorical_cols = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['object', 'str']]
    categorical_cols = df.select_dtypes(['object']).columns

    def chi_test(c):
        print()
        ct = pd.crosstab(df[target_col_str], df[c])
        chi2, p, dof, expected = scs.chi2_contingency(ct)
        exp_under_five_percent = np.sum(expected < 5) / len(expected.flat)

        if exp_under_five_percent > 0.2:
            return None
        else:
            return p

    chi_dict = {c: chi_test(c) for c in categorical_cols}

    chi_df = pd.DataFrame.from_dict(chi_dict, orient='index')
    chi_df.columns = ['p_value']

    chi_df = chi_df.sort_values(by='p_value', ascending=True)

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


def correlation_plot(df, ax):
    """
    Plots the correlation between numerical features in the data
    Use to test for a relationship between each numerical feature
    :param df: Pandas DataFrame
    :param ax: Axes object for figure
    :return: correlations
    """

    # numerical_cols = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['float64', 'int64']]
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numerical_cols].corr()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, cmap=cmap, square=True, annot=True, annot_kws={"size": 10}, linewidths=.5,
                cbar_kws={"shrink": .5}, ax=ax, fmt='.2f')

    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.title('Correlation plot of numerical features')

    return corr


def correlated_features(df, min_corr):
    """
    Takes a DataFrame and returns pairs of numerical features that meet the minimum specified correlation
    :param df: Pandas DataFrame
    :param min_corr: The absolute minimum correlation score to select a feature pair
    :return: Pandas DataFrame with highly correlated feature pairs sorted by absolute(correlation)
    """

    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_cols.remove(target_col_str)
    corr = df[numerical_cols].corr()
    corr_rule = np.where((corr > min_corr) | (corr < -min_corr))

    high_corr_f = [(corr.index[x], corr.columns[y], corr.values[x, y]) for x, y in zip(*corr_rule)
                   if x != y and x < y]

    high_corr_df = pd.DataFrame(high_corr_f, columns=['feature 1', 'feature 2', 'correlation'])
    high_corr_df['abs_corr'] = abs(high_corr_df['correlation'])
    high_corr_df = high_corr_df.sort_values(by='abs_corr', ascending=False)

    return high_corr_df


def point_biserial_corr(df, target_col_str):
    """
    Calculate the point-biserial correlation coefficient for each numerical feature against a dichotomous target
    :param df: Pandas DataFrame
    :param target_col_str: string name of the dichotomous target column (column should be numerical 0,1)
    :return: Pandas DataFrame with the correlation coefficient (r) for each feature, sorted by abs(r) 
    """

    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_cols.remove('target_col_str')
    x = df[target_col_str]

    def pc_corr(c):
        y = df[c]
        r, p = pointbiserialr(x, y)
        return r

    pc_dict = {c: pc(c) for c in numerical_cols}
    pc_df = pd.DataFrame.from_dict(pc_dict, orient='index')
    pc_df.columns = ['corr']
    pc_df['corr_abs'] = abs(pc_df['corr'])
    pc_df = pc_df.sort_values(by='corr_abs', ascending=False)

    return pc_df


def plot_score(df, score_col, n, title, log_scale, score_label):
    """
    Plot values for a single Pandas Series e.g. to visualise results of a statistical test
    :param df: Pandas DataFrame
    :param score_col: str name of column to plot
    :param n: number of values to plot, sort df prior to plotting take top/bottom n
    :param title: str title for plot
    :param log_scale: {True, False} whether a log scale should be used
    :param score_label: str name for axis
    :return: plot object
    """

    plot_s = n * 0.5

    score_plot = df[score_col].head(n).plot(kind='barh', use_index=True, color='deepskyblue',
                                            figsize=(10, plot_s),
                                            fontsize=12,
                                            title=title)

    plt.xlabel(score_label)
    plt.grid(True, axis='x')
    plt.grid(False, axis='y')
    # plt.xlim(-1, 1)
    plt.gca().invert_yaxis()

    if log_scale == True:
        plt.xscale('log')
    else:
        pass

    plt.show()

    return score_plot
