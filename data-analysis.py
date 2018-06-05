__author__ = "Patrick Buhagiar"

import pandas as pd
from matplotlib.pylab import rcParams

from toolbox import plot, load_indices, test_stationarity, full_scatter_plot, make_indices_stationary

rcParams['figure.figsize'] = 14, 5

# Define start and end dates here
start = pd.datetime(1998, 01, 02)
end = pd.datetime(2018, 01, 02)

# Load data and plot data
ts_indices = load_indices(start, end)
plot(ts_indices, "Plot of All Indices")

ts_log_shift = make_indices_stationary(ts_indices)
stationary = {}
for k, v in ts_log_shift.iteritems():
    stationary[k] = test_stationarity(v, k, False)

print stationary

# Plot a scatter plot for every combination of stock pairs
full_scatter_plot(ts_log_shift)
