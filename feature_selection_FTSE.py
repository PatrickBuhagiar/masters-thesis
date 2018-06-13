import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA

from toolbox import extract_macroeconomic_data, load_indices, make_indices_stationary, make_time_series_stationary, \
    test_stationarity

# Load Data
start = pd.datetime(2007, 2, 2)
end = pd.datetime(2017, 2, 2)
GB_Balance_of_Trade = extract_macroeconomic_data("data/macroeconomics/GB/GB_BALANCE_OF_TRADE.csv", 222, start, end)
GB_Gdp = extract_macroeconomic_data("data/macroeconomics/GB/GB_GDP.csv", 222, start, end)
GB_Inflation = extract_macroeconomic_data("data/macroeconomics/GB/GB_INFLATION.csv", 69, start, end)
GB_Interest = extract_macroeconomic_data("data/macroeconomics/GB/GB_INTRST.csv", 1, start, end, 'D')
GB_Unemployment = extract_macroeconomic_data("data/macroeconomics/GB/GB_UNEMPLOYMENT.csv", 350, start, end)
indices = load_indices(start, end)


def f(x):
    if x >= 0:
        return 1
    elif x < 0:
        return 0
    else:
        return 0


discrete_FTSE = map(lambda x: f(x), (indices['FTSE'] - indices['FTSE'].shift().fillna(0)))

# Make data stationary
ts_indices = make_indices_stationary(indices)
ts_trdblc = make_time_series_stationary(GB_Balance_of_Trade.iloc[:, 0])
ts_gdp = make_time_series_stationary(GB_Gdp.iloc[:, 0])
ts_infl = make_time_series_stationary(GB_Inflation.iloc[:, 0])
ts_unemp = make_time_series_stationary(GB_Unemployment.iloc[:, 0])
ts_intr = make_time_series_stationary(GB_Interest.iloc[:, 0])

# Assert that data is stationary
stationary = {"Balance of Trade": test_stationarity(ts_trdblc, "Balance of Trade", False),
              "GDP": test_stationarity(ts_gdp, "GDP", False),
              "Inflation Rate": test_stationarity(ts_infl, "Inflation Rate", False),
              "Interest Rate": test_stationarity(ts_intr, "Interest Rate", False),
              "Unemployment Rate": test_stationarity(ts_unemp, "Unemployment Rate", False)}
for k, v in ts_indices.iteritems():
    stationary[k] = test_stationarity(v, k, False)

print(stationary)

# Gather all data into a single data frame
columns = ['GB_TRADE_BLNC', 'GB_GDP', 'GB_INFL', 'GB_INTR', 'GB_UNEMP', 'CAC', 'DAX', 'HKSE', 'NIKKEI',
           'S&P500', 'STOXX', 'FTSE']
df = pd.DataFrame(index=ts_indices['FTSE'].index, columns=columns)
df['GB_TRADE_BLNC'] = ts_trdblc
df['GB_GDP'] = ts_gdp
df['GB_INFL'] = ts_infl
df['GB_INTR'] = ts_intr
df['GB_UNEMP'] = ts_unemp
df['CAC'] = ts_indices['CAC']
df['DAX'] = ts_indices['DAX']
df['HKSE'] = ts_indices['HKSE']
df['NIKKEI'] = ts_indices['NIKKEI']
df['S&P500'] = ts_indices['S&P500']
df['STOXX'] = ts_indices['STOXX']
df['FTSE'] = ts_indices['FTSE']

# Perform Recursive Feature Elimination
array = df.values
X = array[:, 0:11]
del discrete_FTSE[-1]
Y = np.asarray(discrete_FTSE)
model = LogisticRegression()
rfe = RFE(model, 4)
fit = rfe.fit(X, Y)

print("Num Features: %d") % fit.n_features_
print(columns[:-1])
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_

# Perform Feature Importance

model = ExtraTreesRegressor()
Y = array[:, 11]
model.fit(X, Y)
print (model.feature_importances_)

model = ExtraTreesClassifier()
Y = np.asarray(discrete_FTSE)
model.fit(X, Y)
print (model.feature_importances_)

# PCA
pca = PCA(n_components=3)
fit = pca.fit(X)
print(fit.explained_variance_ratio_)
print(fit.components_)

