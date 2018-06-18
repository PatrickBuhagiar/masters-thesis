import pandas as pd
from stock_model import get_model
from joblib import Parallel, delayed
import multiprocessing

start_year = 2006
end_year = 2016
month = [1, 4, 7, 10]
n_cpus = multiprocessing.cpu_count()
models = {}
classes = {}
sessions = {}
feed_dicts = {}


def process(start):
    end = start + pd.DateOffset(years=2)
    print("Working on", start, "to", end)
    model, actual_classes, sess, feed_dict = get_model(start, end)
    models[start.__str__()] = model
    classes[start.__str__()] = actual_classes
    sessions[start.__str__()] = sess
    feed_dicts[start.__str__()] = feed_dict


def generate_dates(start, end, month):
    list = []
    for i in range(end - start):
        for j in range(len(month)):
            list.append(pd.datetime(start + i, month[j], 1))
    return list


def get_models(start, end):
    dates = generate_dates(start, end, month)
    print(dates)
    Parallel(n_jobs=4)(delayed(process)(date) for date in dates)


if __name__ == '__main__':
    print(n_cpus)
    get_models(start_year, end_year)
    print(models)
    print(classes)
    print(sessions)
    print(feed_dicts)
