import pandas as pd
from toolbox import extract_macroeconomic_data, test_stationarity, make_time_series_stationary

start = pd.datetime(1998, 1, 2)
end = pd.datetime(2016, 1, 2)
GB_Balance_of_Trade = extract_macroeconomic_data("data/macroeconomics/GB/GB_BALANCE_OF_TRADE.csv", 222, start, end)
GB_Gdp = extract_macroeconomic_data("data/macroeconomics/GB/GB_GDP.csv", 222, start, end)
GB_Inflation = extract_macroeconomic_data("data/macroeconomics/GB/GB_INFLATION.csv", 69, start, end)
GB_Interest = extract_macroeconomic_data("data/macroeconomics/GB/GB_INTRST.csv", 1, start, end, 'D')
GB_Unemployment = extract_macroeconomic_data("data/macroeconomics/GB/GB_UNEMPLOYMENT.csv", 350, start, end)

ts_gdp = make_time_series_stationary(GB_Gdp.iloc[:, 0])
ts_infl = make_time_series_stationary(GB_Inflation.iloc[:, 0])
ts_unemp = make_time_series_stationary(GB_Unemployment.iloc[:, 0])
ts_intr = make_time_series_stationary(GB_Interest.iloc[:, 0])

stationary = {"Balance of Trade": test_stationarity(GB_Balance_of_Trade.iloc[:, 0], "Balance of Trade", False),
              "GDP": test_stationarity(ts_gdp, "GDP", False),
              "Inflation Rate": test_stationarity(ts_infl, "Inflation Rate", False),
              "Interest Rate": test_stationarity(ts_intr, "Interest Rate", False),
              "Unemployment Rate": test_stationarity(ts_unemp, "Unemployment Rate", False)}
