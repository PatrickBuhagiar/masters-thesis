import pandas as pd
from toolbox import extract_macroeconomic_data

data = extract_macroeconomic_data("data/macroeconomics/GB/GB_BALANCE_OF_TRADE.csv", 221, pd.datetime(1998, 1, 2), pd.datetime(2017, 12, 31))
print "hello"
