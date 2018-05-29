import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15, 6

from statsmodels.tsa.stattools import adfuller

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv('data/indices/^FCHI.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse)

print data