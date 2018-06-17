import pandas as pd
from stock_model import get_model

start_year = 2006
end_year = 2016
month = [1, 4, 7, 10]


def get_models(start_year, end_year, month):
    dict = {}
    for i in range(end_year - start_year):
        for j in range(len(month)):
            start = pd.datetime(start_year + i, month[j], 1)
            end = pd.datetime(start_year + i + 2, month[j], 1)
            dict[start.__str__()] = get_model(start, end)

    return dict


print(get_models(start_year, end_year, month))
