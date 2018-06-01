__author__ = "Patrick Buhagiar"

import pandas as pd
from matplotlib.pylab import rcParams

from toolbox import normalise, plot, load_indices, test_stationarity, log_transform, rolling_moving_averages, \
    log_moving_averages_diff, differencing, full_scatter_plot

rcParams['figure.figsize'] = 14, 5

# Define start and end dates here
start = pd.datetime(1998, 01, 02)
end = pd.datetime(2018, 01, 02)

# Load data and plot data
ts_indices = load_indices(start, end)
plot(ts_indices, "Plot of All Indices")

# Normalise and plot data
ts_indices_normalised = normalise(ts_indices)
plot(ts_indices_normalised, "Plot of All Normalised Indices")

# Let's make the data stationary
ts_log_indices = log_transform(ts_indices_normalised)
ts_moving_averages = rolling_moving_averages(ts_log_indices, 365)
ts_log_moving_averages_diff = log_moving_averages_diff(ts_log_indices, ts_moving_averages)

# It has been found that differencing is a very good way for this data to be converted into stationary
ts_log_shift = differencing(ts_log_indices)

stationary = {}
for k, v in ts_log_shift.iteritems():
    stationary[k] = test_stationarity(v, k, False)

print stationary

# Plot a scatter plot for every combination of stock pairs
full_scatter_plot(ts_log_shift)
