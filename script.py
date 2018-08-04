import matplotlib.pylab as plt
import pandas as pd
from pandas import Series
import numpy as np

start = pd.datetime(1958, 1, 1)
end = pd.datetime(2018, 1, 1)

from toolbox import convert_to_date, extract_index

dateparse1 = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv("data/macroeconomics/US_GDP.csv", index_col='Date', date_parser=dateparse1)
d = {'Date': [], 'Value': []}

for index, row in data.iterrows():
    d['Date'].append(index)
    d['Value'].append(float(row['Value']))

dateparse2 = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y')
ts_SP = extract_index('data/indices/^GSPC.csv', start, end, dateparse2)


def convert_to_quarterly(ts: Series):
    months = [1, 4, 7, 10]
    d = {'Date': [], 'Value': []}
    for row in ts.iteritems():
        if months.__contains__(row[0].month) and row[0].day == 1:
            d['Date'].append(row[0])
            d['Value'].append(row[1])
    return pd.DataFrame(d)


def log(ts: pd.DataFrame):
    for index, row in ts.iterrows():
        row.Value = row.Value / max(ts.values)


sp = convert_to_quarterly(ts_SP)
sp = sp.set_index('Date')
log(sp)

ts = pd.DataFrame(d)
ts = ts.set_index('Date')
ts = ts[pd.datetime(1958, 1, 1):]
log(ts)

plt.plot(ts, label="U.S. GDP")
plt.plot(sp, label="S&P 500")
plt.legend(loc='best')
plt.title("Plot of U.S. GDP vs S&P 500 index")
plt.show()
