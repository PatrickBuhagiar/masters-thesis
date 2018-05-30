import matplotlib.pylab as plt
import pandas as pd
from matplotlib.pylab import rcParams
import numpy as np

rcParams['figure.figsize'] = 14, 5
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA


# a functions that tests whether a given time series is stationary
def test_stationarity(timeseries, name):
    # Determine rolling statistics
    rolmean = timeseries.rolling(365).mean()  # 365 i.e. for the last year
    rolstd = timeseries.rolling(365).std()

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title("Rolling mean & Standard Deviation for " + name)
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print '\nResults of Dickey-Fuller Test for ' + name + ':'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print dfoutput
    plt.show()


# Load data and parse date
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
dateparse2 = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y')
dateparse3 = lambda dates: pd.datetime.strptime(dates, '%m/%d/%y')

data_CAC = pd.read_csv('data/indices/^FCHI.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse)
ts_CAC = data_CAC['Close']
ts_CAC = ts_CAC.ix['1998-01-02':'2018-01-02']
ts_CAC = ts_CAC[~ts_CAC.isnull()]

data_DAX = pd.read_csv('data/indices/^GDAXI.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse)
ts_DAX = data_DAX['Close']
ts_DAX = ts_DAX.ix['1998-01-02':'2018-01-02']
ts_DAX = ts_DAX[~ts_DAX.isnull()]

data_GSPC = pd.read_csv('data/indices/^GSPC.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse2)
ts_GSPC = data_GSPC['Close']
ts_GSPC = ts_GSPC.ix['02/01/1998':'02/01/2018']
ts_GSPC = ts_GSPC[~ts_GSPC.isnull()]

data_N225 = pd.read_csv('data/indices/^N225.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse)
ts_N225 = data_N225['Close']
ts_N225 = ts_N225.ix['1998-01-02':'2018-01-02']
ts_N225 = ts_N225[~ts_N225.isnull()]

data_STOXX = pd.read_csv('data/indices/^STOXX50E.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse2)
ts_STOXX = data_STOXX['Close']
ts_STOXX = ts_STOXX.ix['02/01/1998':'02/01/2018']
ts_STOXX = ts_STOXX[~ts_STOXX.isnull()]

data_FTSE = pd.read_csv('data/indices/^FTSE.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse3)
ts_FTSE = data_FTSE[' Close']
ts_FTSE = ts_FTSE.ix['2018-01-02':'1998-01-02']
ts_FTSE = ts_FTSE[~ts_FTSE.isnull()]

data_HSI = pd.read_csv('data/indices/^HSI.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse)
ts_HSI = data_HSI['Close']
ts_HSI = ts_HSI.ix['1998-01-02':'2018-01-02']
ts_HSI = ts_HSI[~ts_HSI.isnull()]

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

# Test the stationarity.
# From these results we can see that they are not stationary.
# Check out the printed 'Test Statistic'.
# Ideally it should be less than 1% critical value
# test_stationarity(ts_CAC, 'CAC')
# test_stationarity(ts_DAX, 'DAX')
# test_stationarity(ts_GSPC, 'GSPC')
# test_stationarity(ts_N225, 'N225')
# test_stationarity(ts_STOXX, 'STOXX')
# test_stationarity(ts_FTSE, 'FTSE')
# test_stationarity(ts_HSI, 'HSI')

# Let's make the data stationary
ts_log_CAC = np.log(ts_CAC)
moving_avg = ts_log_CAC.rolling(365).mean()
ts_log_moving_avg_diff = ts_log_CAC - moving_avg
ts_log_moving_avg_diff.dropna(inplace=True)

expweighted_average_cac = ts_log_CAC.ewm(halflife=365).mean()
ts_log_ewma_diff = ts_log_CAC - expweighted_average_cac
test_stationarity(ts_log_ewma_diff, 'CAC exp weigth')
plt.show()

ts_log_cac_diff = ts_log_CAC - ts_log_CAC.shift()
ts_log_cac_diff.dropna(inplace=True)
test_stationarity(ts_log_cac_diff, 'CAC shift')
plt.show()

# ACF and PACF
lag_acf_cac = acf(ts_log_cac_diff, nlags=40)
lag_pacf_cac = pacf(ts_log_cac_diff, nlags=40, method='ols')

# Plot ACF
plt.subplot(121)
plt.plot(lag_acf_cac)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_cac_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_cac_diff)),linestyle='--',color='gray')
plt.title('CAC Autocorrelation Function')

# Plot PACF
plt.subplot(122)
plt.plot(lag_pacf_cac)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_cac_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_cac_diff)),linestyle='--',color='gray')
plt.title('CAC Partial Autocorrelation Function')
plt.tight_layout()

plt.show()

# ARIMA
model = ARIMA(ts_log_CAC, order=(1,1,1))
results_AR_CAC = model.fit(disp=1)
predictions_ARIMA_diff_cac = pd.Series(results_AR_CAC.fittedvalues, copy=True).cumsum()
predictions_ARIMA_log_CAC = pd.Series(ts_log_CAC.ix[0], index=ts_log_CAC.index)
predictions_ARIMA_log_CAC = predictions_ARIMA_log_CAC.add(predictions_ARIMA_diff_cac, fill_value=0)
predictions_CAC = np.exp(predictions_ARIMA_log_CAC)
plt.plot(ts_CAC)
plt.plot(predictions_CAC)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_CAC-ts_CAC)**2)/len(ts_CAC)))
plt.show()