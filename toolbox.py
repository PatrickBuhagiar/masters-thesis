__author__ = "Patrick Buhagiar"

import matplotlib.pylab as plt
import pandas as pd
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 14, 5
from statsmodels.tsa.stattools import adfuller


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
    ts = data['Close']
    ts = ts.ix[start:end]
    ts = ts[~ts.isnull()]
    return ts


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
