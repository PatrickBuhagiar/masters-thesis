import matplotlib.pylab as plt
import pandas as pd
from matplotlib.pylab import rcParams
import numpy as np

rcParams['figure.figsize'] = 14, 5
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from toolbox import test_stationarity, extract_index

# Load data and parse date
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
dateparse2 = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y')
dateparse3 = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')

start = '1998-01-02'
end = '2018-01-02'
ts_CAC = extract_index('data/indices/^FCHI.csv', start, end, dateparse)
ts_DAX = extract_index('data/indices/^GDAXI.csv', start, end, dateparse)
ts_GSPC = extract_index('data/indices/^GSPC.csv', start, end, dateparse2)
ts_N225 = extract_index('data/indices/^N225.csv', start, end, dateparse)
ts_STOXX = extract_index('data/indices/^STOXX50E.csv', start, end, dateparse2)
ts_FTSE = extract_index('data/indices/^FTSE.csv', start, end, dateparse3)
ts_HSI = extract_index('data/indices/^HSI.csv', start, end, dateparse)

# Plot all indices
cac, = plt.plot(ts_CAC, label='CAC 40')
dax, = plt.plot(ts_DAX, label='DAX 30')
gspc, = plt.plot(ts_GSPC, label='S&P 500')
n225, = plt.plot(ts_N225, label='Nikkei 225')
stoxx, = plt.plot(ts_STOXX, label='STOXX 50')
ftse, = plt.plot(ts_FTSE, label='FTSE 100')
hsi, = plt.plot(ts_HSI, label='HKSE')
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
hsi, = plt.plot(ts_HSI, label='HKSE')
plt.legend(handles=[cac, dax, gspc, n225, stoxx, ftse, hsi])
plt.show()

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
plt.axhline(y=-1.96 / np.sqrt(len(ts_log_cac_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96 / np.sqrt(len(ts_log_cac_diff)), linestyle='--', color='gray')
plt.title('CAC Autocorrelation Function')

# Plot PACF
plt.subplot(122)
plt.plot(lag_pacf_cac)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96 / np.sqrt(len(ts_log_cac_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96 / np.sqrt(len(ts_log_cac_diff)), linestyle='--', color='gray')
plt.title('CAC Partial Autocorrelation Function')
plt.tight_layout()

plt.show()

# ARIMA
model = ARIMA(ts_log_CAC, order=(1, 1, 1))
results_AR_CAC = model.fit(disp=1)
predictions_ARIMA_diff_cac = pd.Series(results_AR_CAC.fittedvalues, copy=True).cumsum()
predictions_ARIMA_log_CAC = pd.Series(ts_log_CAC.ix[0], index=ts_log_CAC.index)
predictions_ARIMA_log_CAC = predictions_ARIMA_log_CAC.add(predictions_ARIMA_diff_cac, fill_value=0)
predictions_CAC = np.exp(predictions_ARIMA_log_CAC)
plt.plot(ts_CAC)
plt.plot(predictions_CAC)
plt.title('RMSE: %.4f' % np.sqrt(sum((predictions_CAC - ts_CAC) ** 2) / len(ts_CAC)))
plt.show()
