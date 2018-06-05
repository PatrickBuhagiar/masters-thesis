import pandas as pd
import numpy as ny
from toolbox import extract_macroeconomic_data, load_indices, make_indices_stationary
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

start = pd.datetime(2007, 2, 2)
end = pd.datetime(2017, 2, 2)
GB_Balance_of_Trade = extract_macroeconomic_data("data/macroeconomics/GB/GB_BALANCE_OF_TRADE.csv", 222, start, end)
GB_Gdp = extract_macroeconomic_data("data/macroeconomics/GB/GB_GDP.csv", 222, start, end)
GB_Inflation = extract_macroeconomic_data("data/macroeconomics/GB/GB_INFLATION.csv", 69, start, end)
GB_Interest = extract_macroeconomic_data("data/macroeconomics/GB/GB_INTRST.csv", 1, start, end, 'D')
GB_Unemployment = extract_macroeconomic_data("data/macroeconomics/GB/GB_UNEMPLOYMENT.csv", 350, start, end)
ts_indices = make_indices_stationary(load_indices(start, end))

columns = ['GB_TRADE_BLNC', 'GB_GDP', 'GB_INFL', 'GB_INTR', 'GB_UNEMP', 'CAC', 'DAX', 'HKSE', 'NIKKEI',
           'S&P500', 'STOXX', 'FTSE']
df = pd.DataFrame(index=ts_indices['FTSE'].index, columns=columns)
df['GB_TRADE_BLNC'] = GB_Balance_of_Trade
df['GB_GDP'] = GB_Gdp
df['GB_INFL'] = GB_Inflation
df['GB_INTR'] = GB_Interest
df['GB_UNEMP'] = GB_Unemployment
df['CAC'] = ts_indices['CAC']
df['DAX'] = ts_indices['DAX']
df['HKSE'] = ts_indices['HKSE']
df['NIKKEI'] = ts_indices['NIKKEI']
df['S&P500'] = ts_indices['S&P500']
df['STOXX'] = ts_indices['STOXX']
df['FTSE'] = ts_indices['FTSE']

array = df.values
X = array[:]
Y = array[:, -1]

# test = SelectKBest(score_func=chi2, k=4)
# fit = test.fit(X, Y)
#
# ny.set_printoptions(precision=3)
# print fit.scores_
# features = fit.transform(X)

print "hello"
