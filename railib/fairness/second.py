import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as RegressionModel
from sklearn.metrics import mean_squared_error
from scipy.stats import percentileofscore

sns.set()


def data_statistics(df: DataFrame):
    """
    
    Generates descriptive statistics for each column: 
    count (non NaN values), mean, standard deviation, minimum & maximum values,
    25th, 50th (median), 75th percentiles.
    """
    return df.describe().T.round(2)


def column_distribution(df: DataFrame, x_column: str,
                        group_column: str = None, cumulative: bool = False,
                        is_count: bool = False):
    """
    
    This function plots the distribution (probability histogram) of x_column.
    Then it splits the plot into sub-plots according to the group_column.
    
    * If group_column is not given (or None) it doesn't split.
    * If x_column is a cost column (x_column contains 'cost'), it uses log.
    * If cumulative=True it presents the cumulative probability.
    Note: sometimes it is more easy to see the differences between two variables/values using the cumulative probabilities histogram.
    * If is_count=True it present the number of patients at each bin (count) and not the probability.
    """
    bins = 35
    df = df.copy()
    if 'cost' in x_column:
        df = df.assign(**{x_column: np.log(df[x_column] + 1)})
    if 'risk' in x_column:
        df[x_column] = df[x_column].rank(pct=True).round(2)
        bins = np.arange(0, 1.01, 0.05)

    if group_column is not None:
        groups_order = sorted(df[group_column].unique().tolist())
    else:
        groups_order = None
    sns.histplot(data=df, x=x_column, hue=group_column,
                 hue_order=groups_order, bins=bins,
                 stat='count' if is_count else 'probability',
                 common_norm=False, cumulative=cumulative)
    plt.show()
    

def preprocessing(df: DataFrame, X_columns: list, y_column: str):
    """
    This function preprocesses the dataframe by:
    1. transforming string age column into binary columns for each age group
    2. transforming gender string column into binary column (1 for female)
    3. transforming race string column into binary column (1 for black)
    4. using log on y_column if it is a cost column
    Then it returns X_data and y_data (which are used to fit the ML model)
    """
    X_columns = X_columns.copy()
    # transforms string age column into binary columns for each age group
    if 'age' in X_columns:
        X_columns.remove('age')
        for age_group in df['age'].unique():
            new_column = f'age_{age_group}'
            df = df.assign(**{new_column: (df['age'] == age_group) * 1})
            X_columns.append(new_column)
    # transforms gender string column into binary column (1 for female)
    if 'gender' in X_columns:
        X_columns.remove('gender')
        df = df.assign(**{'female': (df['gender'] == 'Female') * 1})
        X_columns.append('female')
    # transforms race string column into binary column (1 for black)
    if 'race' in X_columns:
        X_columns.remove('race')
        df = df.assign(**{'black': (df['race'] == 'black') * 1})
        X_columns.append('black')
    # uses log on y_column if it is a cost column
    if 'cost' in y_column:
        new_column = f'log_{y_column}'
        df = df.assign(**{new_column: np.log(df[y_column] + 1)})
        y_column = new_column
    X_data, y_data = df[X_columns], df[y_column].rank(pct=True).round(2) * 100
    return X_data, y_data

def predict_risk_score(train: DataFrame, test: DataFrame,
                       X_columns: list, y_column: str):
    """
    * This function first preprocesses the train and test data using preprocessing().
    * Then it uses a Linear Regression model: the features are X_columns and the target column (Y) is the y_column.
    * Then the function uses the model to predict the y of the train and test data sets.
    * After that, the function calculates the risk score of each patient is based on the percentile he belongs to according to the modelpredictions (percentiles are important for the company).
    * The function prints the MSE of the model on the train and the test data.
    * The function returns train_risk_score and test_risk_score.
    """
    X_train, y_train = preprocessing(train, X_columns, y_column)
    X_test, y_test = preprocessing(test, X_columns, y_column)
    
    assert set(X_train.columns) == set(X_test.columns)
    X_test = X_test[X_train.columns]
    
    # fit predict
    model = RegressionModel()
    model.fit(X_train, y_train)
    print("Finished training the model.")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # convert the predictions into percentiles
    train_risk_scores = [percentileofscore(train_pred, p, 'strict')
                        for p in train_pred]
    test_risk_scores = [percentileofscore(train_pred, p, 'strict')
                        for p in test_pred]
    # convert the real y into percentiles
    train_real_percentiles = [percentileofscore(y_train, p, 'strict')
                             for p in y_train]
    test_real_percentiles = [percentileofscore(y_train, p, 'strict')
                            for p in y_test]
    # evaluate the model
    print("Evaluating the model:")
    print("\t Train Mean Squared Error (MSE): "
          f"{mean_squared_error(train_real_percentiles, train_risk_scores)}")
    print("\t Test Mean Squared Error (MSE): "
          f"{mean_squared_error(test_real_percentiles, test_risk_scores)}")
    return train_risk_scores, test_risk_scores
  
  
  # these functions are taken from the pre-class task

def plot_x_vs_y(df: DataFrame, x_column: str, y_column: str,
                group_column: str=None):
    """
    
    This function creates a chart with the following plots:
    1. Line plot of the pairs:
        {(x_column i-th decile, y_column): i = 1, 2, ..., 10}
    2. scatter plot ('X' markers) of the pairs:
        {(x_column j-th percentile, y_column): j = 1, 2, ..., 100}
    3. Vertical dashed lines of 55th and 97th x_column percentiles.
    
    * The function splits each plot into sub-plots according to the group_column.
    * If group_column is not given (or None) it doesn't split the plots.
    * If y_column is a cost column (y_column contains 'cost'), it uses log scale.
    """
    df = df.copy()
    df[f'{x_column} (deciles)'] = df[x_column].rank(pct=True).round(1) * 100
    df[f'{x_column} (percentiles)'] = df[x_column].rank(pct=True).round(2) * 100
    groupby = [f'{x_column} (percentiles)']
    if group_column is not None:
        groupby = [group_column, f'{x_column} (percentiles)']
        groups_order = sorted(df[group_column].unique().tolist())
    else:
        groupby = [f'{x_column} (percentiles)']
        groups_order = None
    x_p_vs_y = df.groupby(groupby)[y_column].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(data=df, x=f'{x_column} (deciles)', y=y_column,
                 hue=group_column, hue_order=groups_order, ax=ax)
    sns.scatterplot(data=x_p_vs_y, x=f'{x_column} (percentiles)',
                    y=y_column, hue=group_column,
                    marker='X', hue_order=groups_order, legend=False, ax=ax)
    plt.axvline(x=55, color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=97, color='k', linestyle='--', alpha=0.5)
    if 'cost' in y_column.lower():
        plt.yscale('log')
    plt.xlabel(f'{x_column} percentile')
    plt.show()


def at_risk_boxplot(df: DataFrame, score_column: str, percentile_threshold: int,
                    y_column: str, group_column: str=None):
    """
    
    This function compares patients at risk against patients not at risk.
    The risk is detemined using the score_column (e.g. risk_score) and a percentile_threshold (integer between 1 and 100). 
    Any patient with score_column >= percentile_threshold is considered at risk.
    
    * The function plots boxplot of y_column for each risk group.
    * The function splits the plot into sub-plots according to the group_column.
    * If group_column is not given (or None) it doesn't split.
    * If y_column is a cost column (y_column contains 'cost'), it uses log scale.
    """
    columns = [score_column, y_column]
    if group_column is not None:
        columns.append(group_column)
        groups_order = sorted(df[group_column].unique().tolist())       
    else:
        groups_order = None
    df = df[columns].copy()
    df['risk_percentile'] = df[score_column].rank(pct=True).round(2) * 100
    df['at_risk'] = df['risk_percentile'] >= percentile_threshold
    sns.boxplot(data=df, x='at_risk', y=y_column, hue=group_column,
                hue_order=groups_order, showfliers=False)
    if 'cost' in y_column.lower():
        plt.yscale('log')
    plt.show()
