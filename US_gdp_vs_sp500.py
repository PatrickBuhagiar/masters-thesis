import matplotlib.pylab as plt
import pandas as pd
from pandas import Series

start = pd.datetime(1958, 1, 1)
end = pd.datetime(2018, 1, 1)

dateparse1 = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv("data/macroeconomics/US_GDP.csv", index_col='Date', date_parser=dateparse1)
d = {'Date': [], 'Value': []}


def extract_index(filename, start, end, date_parse, dropna=True):
    """
    Extracts the index from a csv file and filters base_out into a date range.

    :param  filename: The name of the csv file
    :param     start: The start date
    :param       end: the end date
    :param date_parse: the type of date parsing
    :param dropna: drop any nas

    :return: The indices as a time series
    """
    data = pd.read_csv(filename, parse_dates=['Date'], index_col='Date', date_parser=date_parse)
    # Fill missing dates and values
    all_days = pd.date_range(start, end, freq='D')
    data = data.reindex(all_days)
    ts = data['Close']
    if dropna:
        ts = ts.dropna()
    return ts


for index, row in data.iterrows():
    d['Date'].append(index)
    d['Value'].append(float(row['Value']))

dateparse2 = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y')
ts_SP = extract_index('data/indices/^GSPC.csv', start, end, dateparse2)


def convert_to_quarterly(ts: Series):
    """
    Convert time series to quarterly data
    :param ts:
    :return:
    """
    months = [1, 4, 7, 10]
    d = {'Date': [], 'Value': []}
    for row in ts.iteritems():
        if months.__contains__(row[0].month) and row[0].day == 1:
            d['Date'].append(row[0])
            d['Value'].append(row[1])
    return pd.DataFrame(d)


def normalise(ts: pd.DataFrame):
    """
    normalise the time series
    :param ts:
    :return:
    """
    for index, row in ts.iterrows():
        row.Value = row.Value / max(ts.values)


sp = convert_to_quarterly(ts_SP)
sp = sp.set_index('Date')
normalise(sp)

ts = pd.DataFrame(d)
ts = ts.set_index('Date')
ts = ts[pd.datetime(1958, 1, 1):]
normalise(ts)

plt.plot(ts, label="U.S. GDP")
plt.plot(sp, label="S&P 500")
plt.legend(loc='best')
plt.title("Plot of U.S. GDP vs S&P 500 index")
plt.show()
