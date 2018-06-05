import pandas as pd
from toolbox import extract_macroeconomic_data

start = pd.datetime(1998, 1, 2)
end = pd.datetime(2016, 1, 2)
GB_Balance_of_Trade = extract_macroeconomic_data("data/macroeconomics/GB/GB_BALANCE_OF_TRADE.csv", 222, start, end)
GB_GDP = extract_macroeconomic_data("data/macroeconomics/GB/GB_GDP.csv", 222, start, end)
GB_INFLATION = extract_macroeconomic_data("data/macroeconomics/GB/GB_INFLATION.csv", 70, start, end)
print "hello"
