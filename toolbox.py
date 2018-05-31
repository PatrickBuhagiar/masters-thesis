__author__ = "Patrick Buhagiar"

import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 14, 5
from statsmodels.tsa.stattools import adfuller


def load_indices(start, end):
    """
    Predefined function that loads all the indices from CSV into a python dictionary.

    :param start: the start date
    :param end: the end date
    :return: a python dictionary of all the indices
    """
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    dateparse2 = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y')
    dateparse3 = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')
    ts_CAC = extract_index('data/indices/^FCHI.csv', start, end, dateparse)
    ts_DAX = extract_index('data/indices/^GDAXI.csv', start, end, dateparse)
    ts_GSPC = extract_index('data/indices/^GSPC.csv', start, end, dateparse2)
    ts_N225 = extract_index('data/indices/^N225.csv', start, end, dateparse)
    ts_STOXX = extract_index('data/indices/^STOXX50E.csv', start, end, dateparse2)
    ts_FTSE = extract_index('data/indices/^FTSE.csv', start, end, dateparse3)
    ts_HSI = extract_index('data/indices/^HSI.csv', start, end, dateparse)
    return {"CAC": ts_CAC, "DAX": ts_DAX, "S&P500": ts_GSPC, "NIKKEI": ts_N225, "STOXX": ts_STOXX,
            "FTSE": ts_FTSE,
            "HKSE": ts_HSI}


def extract_index(filename, start, end, date_parse):
    """
    Extracts the index from a csv file and filters out into a date range.

    :param  filename: The name of the csv file
    :param     start: The start date
    :param       end: the end date
    :param date_parse: the type of date parsing

    :return: The indices as a time series
    """
    data = pd.read_csv(filename, parse_dates=['Date'], index_col='Date', date_parser=date_parse)
    # Fill missing dates and values
    all_days = pd.date_range(start,end, freq='D')
    data = data.reindex(all_days)
    data = data.fillna(method='ffill')
    ts = data['Close']
    return ts


def plot(indices, title):
    """
    Plots all the indices on the same plot.

    :param indices: the indices to be plotted
    :param title: the title of the plot
    :return: nothing
    """
    for k, v in indices.iteritems():
        plt.plot(v, label=k)
    plt.legend(loc='best')
    plt.title(title)
    plt.show()


def normalise(indices):
    """
    Normalises all the indices into the range between 0 and 1.

    :param indices: the indices to be normalise.
    :return: normalised indices dictionary.
    """
    return {k: v / max(v) for k, v in indices.items()}


def test_stationarity(time_series, name, plot=True, print_dft=True, time=365):
    """
    A function that tests whether a given time series is stationary
    by performing a Dickey-Fuller test.

    Ideally, the Test Statistic should be smaller than the 1% Critical value in order
    for the time series to be considered stationary.

    :param time_series: The time series that needs to be check for stationarity
    :param        name: The name of the time series
    :param        plot: If True, plot the time series, rolling mean and rolling std. Default is False
    :param   print_dft: If True, print the Dickey-Fuller test results on the console. Default is True
    :param        time: The time needed for rolling mean and standard deviation. Default is 365 i.e. a year

    :return: True if stationary
    """

    # Determine rolling statistics
    rolling_mean = time_series.rolling(time).mean()
    rolling_std = time_series.rolling(time).std()
    df_test = adfuller(time_series, autolag='AIC')
    df_output = pd.Series(df_test[0:4],
                          index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in df_test[4].items():
        df_output['Critical Value (%s)' % key] = value

    if print_dft:
        # Perform Dickey-Fuller test
        print '\nResults of Dickey-Fuller Test for ' + name + ':'
        print df_output

    if plot:
        # Plot rolling statistics:
        plt.plot(time_series, color='blue', label='Original')
        plt.plot(rolling_mean, color='red', label='Rolling Mean')
        plt.plot(rolling_std, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title("Rolling mean & Standard Deviation for " + name)
        plt.show(block=False)
        plt.show()

    return df_output[0] < df_output[5]


def log_transform(indices):
    return {k: np.log(v) for k, v in indices.items()}


def rolling_moving_averages(indices, time):
    return {k: v.rolling(time).mean() for k, v in indices.items()}


def log_moving_averages_diff(log_indices, moving_averages):
    diff = {}
    for k, v in log_indices.iteritems():
        diff[k] = v - moving_averages[k]
        diff[k].dropna(inplace=True)
    return diff


def differencing(indices):
    diff = {}
    for k, v in indices.iteritems():
        diff[k] = v - v.shift()
        diff[k].dropna(inplace=True)
    return diff
