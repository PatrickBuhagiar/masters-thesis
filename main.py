import pandas as pd
from stock_model import get_model
from joblib import Parallel, delayed
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, wait, as_completed

start_year = 2006
end_year = 2014
month = [1, 4, 7, 10]
n_cpus = multiprocessing.cpu_count()
models = {}
classes = {}
sessions = {}
test_dicts = {}

pool = ThreadPoolExecutor(4)
futures = []


def process(start):
    end = start + pd.DateOffset(years=3)
    print("Working on", start, "to", end)
    model, actual_classes, sess, test_dict = get_model(start, end)
    models[start.__str__()] = model
    classes[start.__str__()] = actual_classes
    sessions[start.__str__()] = sess
    test_dicts[start.__str__()] = test_dict


def generate_dates(start, end, month):
    list = []
    for i in range(end - start):
        for j in range(len(month)):
            list.append(pd.datetime(start + i, month[j], 1))
    return list


def get_models(start, end):
    dates = generate_dates(start, end, month)
    print(dates)

    for date in dates:
        futures.append(pool.submit(process, date))

    wait(futures)
    print("PRINTING STUFF")
    print("Models", models)
    print("Classes", classes)
    print("Sessions", sessions)
    print("Test", test_dicts)


if __name__ == '__main__':
    print(n_cpus)
    get_models(start_year, end_year)
