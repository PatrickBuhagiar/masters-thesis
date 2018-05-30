import pandas as pd
import numpy as np
import pandas_datareader.data as data
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from datetime import datetime as dt

rcParams['figure.figsize'] = 15, 6

from statsmodels.tsa.stattools import adfuller

# Load data and parse date
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
dateparse2 = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y')
dateparse3 = lambda dates: pd.datetime.strptime(dates, '%m/%d/%y')

data_CAC = pd.read_csv('data/indices/^FCHI.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse)
ts_CAC = data_CAC['Close']
ts_CAC = ts_CAC.ix['1998-01-02':'2018-01-02']

data_DAX = pd.read_csv('data/indices/^GDAXI.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse)
ts_DAX = data_DAX['Close']
ts_DAX = ts_DAX.ix['1998-01-02':'2018-01-02']

data_GSPC = pd.read_csv('data/indices/^GSPC.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse2)
ts_GSPC = data_GSPC['Close']
ts_GSPC = ts_GSPC.ix['02/01/1998':'02/01/2018']

data_N225 = pd.read_csv('data/indices/^N225.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse)
ts_N225 = data_N225['Close']
ts_N225 = ts_N225.ix['1998-01-02':'2018-01-02']

data_STOXX = pd.read_csv('data/indices/^STOXX50E.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse2)
ts_STOXX = data_STOXX['Close']
ts_STOXX = ts_STOXX.ix['02/01/1998':'02/01/2018']

data_FTSE = pd.read_csv('data/indices/^FTSE.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse3)
ts_FTSE = data_FTSE[' Close']
ts_FTSE = ts_FTSE.ix['2018-01-02':'1998-01-02']

data_HSI = pd.read_csv('data/indices/^HSI.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse)
ts_HSI = data_HSI['Close']
ts_HSI = ts_HSI.ix['1998-01-02':'2018-01-02']

# Plot all indices
cac, = plt.plot(ts_CAC, label='CAC 40')
dax, = plt.plot(ts_DAX, label='DAX 30')
gspc, = plt.plot(ts_GSPC, label='S&P 500')
n225, = plt.plot(ts_N225, label='Nikkei 225')
stoxx, = plt.plot(ts_STOXX, label='STOXX 50')
ftse, = plt.plot(ts_FTSE, label='FTSE 100')
hsi, = plt.plot(ts_FTSE, label='HKSE')
plt.legend(handles=[cac, dax, gspc, n225, stoxx, ftse, hsi])
plt.show()

# Normalisation (range 0 to 1)
ts_CAC = ts_CAC / max(ts_CAC)
ts_DAX = ts_DAX / max(ts_DAX)
ts_GSPC = ts_GSPC / max(ts_GSPC)
ts_N225 = ts_N225 / max(ts_N225)
ts_STOXX = ts_STOXX / max(ts_STOXX)
ts_FTSE = ts_FTSE / max(ts_FTSE)
ts_HSI = ts_HSI / max(ts_HSI)

# Plot all indices
cac, = plt.plot(ts_CAC, label='CAC 40')
dax, = plt.plot(ts_DAX, label='DAX 30')
gspc, = plt.plot(ts_GSPC, label='S&P 500')
n225, = plt.plot(ts_N225, label='Nikkei 225')
stoxx, = plt.plot(ts_STOXX, label='STOXX 50')
ftse, = plt.plot(ts_FTSE, label='FTSE 100')
hsi, = plt.plot(ts_FTSE, label='HKSE')
plt.legend(handles=[cac, dax, gspc, n225, stoxx, ftse, hsi])
plt.show()

start = dt(2016,5,31)
stocks = ['FB', 'AMZN', 'GOOG', 'NFLX', 'GLD']
stocks_data = data.DataReader(stocks, 'google', start)['Close']
stocks_data.describe()