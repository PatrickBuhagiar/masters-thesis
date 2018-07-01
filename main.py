import multiprocessing
from concurrent.futures import ThreadPoolExecutor, wait

import pandas as pd
from pymongo import MongoClient

from stock_model import build_model

# static data
start_year = 2010
end_year = 2011
month = [1, 4, 7, 10]
n_cpus = multiprocessing.cpu_count()

# Concurrency stuff
pool = ThreadPoolExecutor(4)
futures = []

# MongoDB
client = MongoClient('localhost', 27017)
db = client['thesis']


def process(start):
    """
    This is a parallelisable process that builds and stores the model into MongoDB.

    :param start: the start date. The end date is an offset of 3 years
    :return:
    """
    end = start + pd.DateOffset(years=3)
    print("Working on", start, "to", end)
    # build model
    saver, sess, test_dict, train_dict, f1_score, accuracy = build_model(start, end)

    # If model already exists in db, check accuracy and replace if new model is better.
    posts = db.posts
    post = {'_id': start.date().__str__(),
            'test_data': test_dict,
            'train_data': train_dict,
            'f1_score': f1_score,
            'accuracy': accuracy}
    if posts.find_one({'_id': start.date().__str__()}) is not None:
        stored_accuracy = posts.find_one({'_id': start.date().__str__()})['accuracy']
        if stored_accuracy > accuracy:
            print("no need to replace data for", start.date().__str__())
            print("existing stored accuracy", stored_accuracy, "vs", accuracy)
            return
    # All model variables are stored in session
    saver.save(sess, "models/" + start.date().__str__() + "/" + start.date().__str__())
    sess.close()
    posts.update_one({'_id': start.date().__str__()}, {'$set': post}, upsert=True)
    print("Storing!")


def generate_start_dates(start, end):
    """
    generate all the possible start dates that are shifted by a quarter. cascading window.
    :param start: the start date
    :param end: the final (start) date
    :return: a list all the start dates shifted by a quarter
    """
    list = []
    for i in range(end - start):
        for j in range(len(month)):
            list.append(pd.datetime(start + i, month[j], 1))
    return list


def build_models(start, end):
    """
    Build all the models in parallel for all possible start dates
    :param start:
    :param end:
    :return:
    """
    start_dates = generate_start_dates(start, end)
    print("Date Range:", start_dates)

    for start_date in start_dates:
        futures.append(pool.submit(process, start_date))
    wait(futures)


if __name__ == '__main__':
    print("number of CPUs:", n_cpus)
    build_models(start_year, end_year)
