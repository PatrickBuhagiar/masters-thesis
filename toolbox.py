__author__ = "Patrick Buhagiar"

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
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
    all_days = pd.date_range(start - pd.DateOffset(years=1), end, freq='D')
    data = data.reindex(all_days)
    data = data.fillna(method='ffill')
    filtered_days = pd.date_range(start, end, freq='D')
    data = data.reindex(filtered_days)
    data = data.fillna(method='ffill')
    ts = data['Close']
    return ts


def load_macroeconomic_data(filename, start_index, type='Q'):
    data = pd.read_csv(filename, index_col='Date')[start_index:]
    if type == 'D':
        all_days = pd.date_range(data.index[0], data.index[-1])
        data = data.reindex(all_days, fill_value=0)
        data.fillna(method='ffill')
    d = {}

    for date, row in data.iterrows():
        if type == 'Q':
            d[(stringify(extract_quarterly(date)))] = float(row['Value'])
        elif type == 'M':
            d[stringify(extract_month(date))] = float(row['Value'])
        elif type == 'D':
            parsed_date = pd.to_datetime(date)
            d[parsed_date.__str__()] = float(row['Value'])
    return d


def get_lagged_macroeconomic_data(data, date: pd.datetime, type='Q'):
    if type == 'Q':
        # convert current date to Q1 and year, and return that quarter and the 3 previous one
        if 0 < date.month < 4:
            t_0 = data[stringify([date.year, 1])]
            t_1 = data[stringify([date.year - 1, 4])]
            t_2 = data[stringify([date.year - 1, 3])]
            t_3 = data[stringify([date.year - 1, 2])]
        elif 4 <= date.month < 7:
            t_0 = data[stringify([date.year, 2])]
            t_1 = data[stringify([date.year, 1])]
            t_2 = data[stringify([date.year - 1, 4])]
            t_3 = data[stringify([date.year - 1, 3])]
        elif 7 <= date.month < 10:
            t_0 = data[stringify([date.year, 3])]
            t_1 = data[stringify([date.year, 2])]
            t_2 = data[stringify([date.year, 1])]
            t_3 = data[stringify([date.year - 1, 4])]
        else:
            t_0 = data[stringify([date.year, 4])]
            t_1 = data[stringify([date.year, 3])]
            t_2 = data[stringify([date.year, 2])]
            t_3 = data[stringify([date.year, 1])]
        return [t_0, t_1, t_2, t_3]
    elif type == 'M':
        t_0 = data[stringify([date.year, date.month])]
        t_1_date = date - pd.DateOffset(months=3)
        t_1 = data[stringify([t_1_date.year, t_1_date.month])]
        t_2_date = date - pd.DateOffset(months=6)
        t_2 = data[stringify([t_2_date.year, t_2_date.month])]
        t_3_date = date - pd.DateOffset(months=12)
        t_3 = data[stringify([t_3_date.year, t_3_date.month])]
        return [t_0, t_1, t_2, t_3]  # convert that date to month, and return the current month, 3, 6, and 12 months ago
    elif type == 'D':
        t_0 = data.T[date].Value
        t_1 = data.T[(date - pd.DateOffset(months=3)).__str__()].Value
        t_2 = data.T[(date - pd.DateOffset(months=6)).__str__()].Value
        t_3 = data.T[(date - pd.DateOffset(months=12)).__str__()].Value
        return [t_0, t_1, t_2, t_3]


def extract_macroeconomic_data(filename, start_index, start, end, type='Q'):
    """
    Extracts macroeconomic data from a csv file and filters out into a date range.

    :param filename: The name of the csv file
    :param start_index: the start index of the csv file (therefore skip the first n values). In some cases, initial values of csv files are unwanted
    :param start: The start date
    :param end: The end date
    :param type: 'D' for dates, otherwise default is 'Q' which means quarterly, but technically can parse monthly data too

    :return: Daily macroeconomic data as a time series
    """
    data = pd.read_csv(filename, index_col='Date')[start_index:]
    d = {'Date': [], 'Value': []}

    for index, row in data.iterrows():
        if type == 'Q':
            d['Date'].append(convert_to_date(index))
        elif type == 'D':
            d['Date'].append(pd.to_datetime(index))
        d['Value'].append(float(row['Value']))

    ts = pd.DataFrame(d)
    ts = ts.set_index('Date')
    all_days = pd.date_range(start - pd.DateOffset(months=1), end, freq='D')
    ts = ts.reindex(all_days)
    ts = ts.fillna(method='ffill')
    filtered_day_range = pd.date_range(start, end, freq='D')
    ts = ts.reindex(filtered_day_range)
    data.fillna(method='ffill')
    ts = ts.dropna()
    return ts


def stringify(data: []):
    return data.__str__()


def convert_to_date(date):
    """
    Covert a given date String into a pandas datetime

    :param date: A date as a string. this can be in the following formats:
            - 2018Q1
            - 2018 JAN

    :return: a datetime
    """
    if date.__contains__("Q"):
        year, quarter = date.replace(" ", "").split("Q")
        if quarter == "1":
            return pd.datetime(int(year), 1, 1)
        elif quarter == "2":
            return pd.datetime(int(year), 4, 1)
        elif quarter == "3":
            return pd.datetime(int(year), 7, 1)
        else:
            return pd.datetime(int(year), 10, 1)
    else:
        year, month = date.split(" ")
        int_month = 0
        if month == "JAN":
            int_month = 1
        elif month == "FEB":
            int_month = 2
        elif month == "MAR":
            int_month = 3
        elif month == "APR":
            int_month = 4
        elif month == "MAY":
            int_month = 5
        elif month == "JUN":
            int_month = 6
        elif month == "JUL":
            int_month = 7
        elif month == "AUG":
            int_month = 8
        elif month == "SEP":
            int_month = 9
        elif month == "OCT":
            int_month = 10
        elif month == "NOV":
            int_month = 11
        elif month == "DEC":
            int_month = 12
        return pd.datetime(int(year), int_month, 1)


def extract_month(date):
    year, month = date.split(" ")
    int_month = 0
    if month == "JAN":
        int_month = 1
    elif month == "FEB":
        int_month = 2
    elif month == "MAR":
        int_month = 3
    elif month == "APR":
        int_month = 4
    elif month == "MAY":
        int_month = 5
    elif month == "JUN":
        int_month = 6
    elif month == "JUL":
        int_month = 7
    elif month == "AUG":
        int_month = 8
    elif month == "SEP":
        int_month = 9
    elif month == "OCT":
        int_month = 10
    elif month == "NOV":
        int_month = 11
    elif month == "DEC":
        int_month = 12
    return [int(year), int_month]


def extract_quarterly(date):
    if "Q" in date:
        year, quarter = date.replace(" ", "").split("Q")
        if quarter == "1":
            return [int(year), 1]
        elif quarter == "2":
            return [int(year), 2]
        elif quarter == "3":
            return [int(year), 3]
        else:
            return [int(year), 4]


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
        print('\nResults of Dickey-Fuller Test for ' + name + ':')
        print(df_output)

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
    """
    Calculate the log of each value.

    :param indices: the indices
    :return: dictionary of log values
    """
    return {k: np.log(v) for k, v in indices.items()}


def rolling_moving_averages(indices, time):
    """
    Calculate the rolling mean of each index

    :param indices: the indices
    :param time: the length of time
    :return: dictionary of rolling means
    """
    return {k: v.rolling(time).mean() for k, v in indices.items()}


def log_moving_averages_diff(log_indices, moving_averages):
    """
    THe difference between log values and moving averages

    :param log_indices: the log indices
    :param moving_averages: the moving averages
    :return: dictionary of this difference
    """
    diff = {}
    for k, v in log_indices.iteritems():
        diff[k] = v - moving_averages[k]
        diff[k].dropna(inplace=True)
    return diff


def differencing(indices):
    """
    Remove trends and seasonality by shifting the time series and taking the difference.

    :param indices: indices to be shifted
    :return: dictionary of the differences
    """
    diff = {}
    for k, v in indices.iteritems():
        diff[k] = v - v.shift()
        diff[k].dropna(inplace=True)
    return diff


def full_scatter_plot(indices):
    """
    Plot a scatter plot for every combination of the time series. this will result in a grid of n x n scatter plots, 
    where n is the length of the time series dictionary.
    
    :param indices: the dictionary of indices
    :return: 
    """
    i = 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for k, v in indices.iteritems():
        for k2, v in indices.iteritems():
            a = fig.add_subplot(indices.__len__(), indices.__len__(), i)
            a.scatter(indices[k], indices[k2])
            a.spines['top'].set_color('none')
            a.spines['bottom'].set_color('none')
            a.spines['left'].set_color('none')
            a.spines['right'].set_color('none')
            a.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
            a.set_title(k + " vs " + k2)
            i += 1
    plt.show()


def make_indices_stationary(indices, doplot=False):
    """
    Make indices dictionary stationary
    :param indices: the indices
    :param doplot: set to True if you want to plot
    :return: stationary indices
    """
    # Normalise and plot data
    ts_indices_normalised = normalise(indices)
    if doplot:
        plot(ts_indices_normalised, "Plot of All Normalised Indices")

    # Let's make the data stationary
    ts_log_indices = log_transform(ts_indices_normalised)
    ts_moving_averages = rolling_moving_averages(ts_log_indices, 365)
    # ts_log_moving_averages_diff = log_moving_averages_diff(ts_log_indices, ts_moving_averages)

    # It has been found that differencing is a very good way for this data to be converted into stationary
    return differencing(ts_log_indices)


def make_time_series_stationary(ts):
    """
    Make a specific time series stationary
    :param ts: the time series
    :return: stationary time series
    """
    ts_normalised_gdp = ts / max(ts)
    ts_log = np.log(ts_normalised_gdp)
    diff = ts_log - ts_log.shift()
    diff.dropna(inplace=True)
    return diff
