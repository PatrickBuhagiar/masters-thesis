import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from bson.binary import Binary
from pymongo import MongoClient

from best_architecture import tf_confusion_metrics
from toolbox import load_indices


def normalise_stocks(stocks):
    """
    normalise stock data such that they fall in the range 0 to 1

    :param stocks: the stock data
    :return: normalised stock data
    """
    stocks['CAC_scaled'] = stocks['CAC'] / max(stocks['CAC'])
    stocks['DAX_scaled'] = stocks['DAX'] / max(stocks['DAX'])
    stocks['HKSE_scaled'] = stocks['HKSE'] / max(stocks['HKSE'])
    stocks['NIKKEI_scaled'] = stocks['NIKKEI'] / max(stocks['NIKKEI'])
    stocks['S&P500_scaled'] = stocks['S&P500'] / max(stocks['S&P500'])
    stocks['STOXX_scaled'] = stocks['STOXX'] / max(stocks['STOXX'])
    stocks['FTSE_scaled'] = stocks['FTSE'] / max(stocks['FTSE'])


def load_data(start, end):
    """
    Load the stock data in the given date range

    :param start: the start date
    :param end: the end date
    :return: stock data for all markets in given date range
    """
    stock_indices = load_indices(start, end)
    stock_data = pd.DataFrame(index=stock_indices['FTSE'].index)
    stock_data['CAC'] = stock_indices['CAC']
    stock_data['DAX'] = stock_indices['DAX']
    stock_data['HKSE'] = stock_indices['HKSE']
    stock_data['NIKKEI'] = stock_indices['NIKKEI']
    stock_data['S&P500'] = stock_indices['S&P500']
    stock_data['STOXX'] = stock_indices['STOXX']
    stock_data['FTSE'] = stock_indices['FTSE']
    normalise_stocks(stock_data)
    return stock_data


def log_diff(stocks):
    """
    Take the log of the stock difference.

    :param stocks: stock data
    :return: log difference of stock data
    """
    log_return_data = pd.DataFrame()
    log_return_data['CAC_log_return'] = np.log(stocks['CAC'] / stocks['CAC'].shift())
    log_return_data['DAX_log_return'] = np.log(stocks['DAX'] / stocks['DAX'].shift())
    log_return_data['HKSE_log_return'] = np.log(stocks['HKSE'] / stocks['HKSE'].shift())
    log_return_data['NIKKEI_log_return'] = np.log(stocks['NIKKEI'] / stocks['NIKKEI'].shift())
    log_return_data['S&P500_log_return'] = np.log(stocks['S&P500'] / stocks['S&P500'].shift())
    log_return_data['STOXX_log_return'] = np.log(stocks['STOXX'] / stocks['STOXX'].shift())
    log_return_data['FTSE_log_return'] = np.log(stocks['FTSE'] / stocks['FTSE'].shift())
    return log_return_data


def extract_market_directions(data):
    """
    create two additional columns that show the direction of the stock market.
    One column is for rise (or equal), and the other is for fall.

    It is assumed that the stock data already contains log differences
    :param data: the stock data
    :return: two new columns in the stock data with market directions.
    """
    data['ftse_log_return_positive'] = 0
    data.ix[data['FTSE_log_return'] >= 0, 'ftse_log_return_positive'] = 1
    data['ftse_log_return_negative'] = 0
    data.ix[data['FTSE_log_return'] < 0, 'ftse_log_return_negative'] = 1


def organise_data(stocks):
    """
    prepare the stock data as input for the neural network.
    for each market, we are taking three days of data.
    In the case of european markets, we cannot take today's closing data,
    so the three days of data start from the previous day. Otherwise, we start
    with today's closing price.

    :param stocks: the stock data
    :return: prepared data for input into neural network
    """
    data = pd.DataFrame(
        columns=[
            'ftse_log_return_positive', 'ftse_log_return_negative',
            'ftse_log_return_1', 'ftse_log_return_2', 'ftse_log_return_3',
            'cac_log_return_1', 'cac_log_return_2', 'cac_log_return_3',
            'dax_log_return_1', 'dax_log_return_2', 'dax_log_return_3',
            'stoxx_log_return_1', 'stoxx_log_return_2', 'stoxx_log_return_3',
            'hkse_log_return_0', 'hkse_log_return_1', 'hkse_log_return_2',
            'nikkei_log_return_0', 'nikkei_log_return_1', 'nikkei_log_return_2',
            's&p500_log_return_0', 's&p500_log_return_1', 's&p500_log_return_2'
        ]
    )
    for i in range(7, len(stocks)):
        ftse_log_return_positive = stocks['ftse_log_return_positive'].ix[i]
        ftse_log_return_negative = stocks['ftse_log_return_negative'].ix[i]
        ftse_log_return_1 = stocks['FTSE_log_return'].ix[i - 1]
        ftse_log_return_2 = stocks['FTSE_log_return'].ix[i - 2]
        ftse_log_return_3 = stocks['FTSE_log_return'].ix[i - 3]
        cac_log_return_1 = stocks['CAC_log_return'].ix[i - 1]
        cac_log_return_2 = stocks['CAC_log_return'].ix[i - 2]
        cac_log_return_3 = stocks['CAC_log_return'].ix[i - 3]
        dax_log_return_1 = stocks['DAX_log_return'].ix[i - 1]
        dax_log_return_2 = stocks['DAX_log_return'].ix[i - 2]
        dax_log_return_3 = stocks['DAX_log_return'].ix[i - 3]
        stoxx_log_return_1 = stocks['STOXX_log_return'].ix[i - 1]
        stoxx_log_return_2 = stocks['STOXX_log_return'].ix[i - 2]
        stoxx_log_return_3 = stocks['STOXX_log_return'].ix[i - 3]
        hkse_log_return_0 = stocks['HKSE_log_return'].ix[i]
        hkse_log_return_1 = stocks['HKSE_log_return'].ix[i - 1]
        hkse_log_return_2 = stocks['HKSE_log_return'].ix[i - 2]
        nikkei_log_return_0 = stocks['NIKKEI_log_return'].ix[i]
        nikkei_log_return_1 = stocks['NIKKEI_log_return'].ix[i - 1]
        nikkei_log_return_2 = stocks['NIKKEI_log_return'].ix[i - 2]
        sp500_log_return_0 = stocks['S&P500_log_return'].ix[i]
        sp500_log_return_1 = stocks['S&P500_log_return'].ix[i - 1]
        sp500_log_return_2 = stocks['S&P500_log_return'].ix[i - 2]

        data = data.append(
            {
                'ftse_log_return_positive': ftse_log_return_positive,
                'ftse_log_return_negative': ftse_log_return_negative,
                'ftse_log_return_1': ftse_log_return_1,
                'ftse_log_return_2': ftse_log_return_2,
                'ftse_log_return_3': ftse_log_return_3,
                'cac_log_return_1': cac_log_return_1,
                'cac_log_return_2': cac_log_return_2,
                'cac_log_return_3': cac_log_return_3,
                'dax_log_return_1': dax_log_return_1,
                'dax_log_return_2': dax_log_return_2,
                'dax_log_return_3': dax_log_return_3,
                'stoxx_log_return_1': stoxx_log_return_1,
                'stoxx_log_return_2': stoxx_log_return_2,
                'stoxx_log_return_3': stoxx_log_return_3,
                'hkse_log_return_0': hkse_log_return_0,
                'hkse_log_return_1': hkse_log_return_1,
                'hkse_log_return_2': hkse_log_return_2,
                'nikkei_log_return_0': nikkei_log_return_0,
                'nikkei_log_return_1': nikkei_log_return_1,
                'nikkei_log_return_2': nikkei_log_return_2,
                's&p500_log_return_0': sp500_log_return_0,
                's&p500_log_return_1': sp500_log_return_1,
                's&p500_log_return_2': sp500_log_return_2
            }, ignore_index=True
        )
    return data


def divide_into_training_testing(inputs, outputs, n):
    """
    Divide the data into training and testing.
    This is split as 80/20.

    :param inputs: the input data
    :param outputs: the output data
    :param n: the size of the dataset (training + testing)
    :return: the inputs and outputs of both the training and testing data
    """
    training_set_size = int(n * 0.8)  # 80/20 sep of training/testing
    training_inputs = inputs[:training_set_size]
    training_outputs = outputs[:training_set_size]
    test_inputs = inputs[training_set_size:]
    test_outputs = outputs[training_set_size:]
    return test_outputs, test_inputs, training_outputs, training_inputs


def build_model(start, end):
    """
    Build a model for a given period.

    The chosen architecture is one with one hidden layer that contains 45 nodes.

    :param start: the start date
    :param end: the end date
    :return: the model, test and training dictionaries, accuracy and F1 score
    """
    n_hidden_nodes = 45

    # Load and prepare data
    df = load_data(start, end)
    log_return_data = log_diff(df)
    extract_market_directions(log_return_data)
    training_test_data = organise_data(log_return_data)

    # Split data into training and testing
    inputs = training_test_data[training_test_data.columns[2:]]
    outputs = training_test_data[training_test_data.columns[:2]]
    test_outputs, test_inputs, training_outputs, training_inputs = divide_into_training_testing(
        inputs, outputs, len(training_test_data))

    # Construct Tensorflow model

    sess = tf.Session()
    num_predictors = len(training_inputs.columns)
    num_classes = len(training_outputs.columns)
    feature_data = tf.placeholder("float", [None, num_predictors], name="feature_data")
    actual_classes = tf.placeholder("float", [None, num_classes], name="actual_classes")

    weights1 = tf.Variable(tf.truncated_normal([num_predictors, n_hidden_nodes], stddev=0.0001), name="w1")
    biases1 = tf.Variable(tf.ones([n_hidden_nodes]), name="b1")
    weights2 = tf.Variable(tf.truncated_normal([n_hidden_nodes, num_classes], stddev=0.0001), name="w2")
    biases2 = tf.Variable(tf.ones([2]), name="b2")

    hidden_layer = tf.nn.relu(tf.matmul(feature_data, weights1) + biases1, name="h")
    model = tf.nn.softmax(tf.matmul(hidden_layer, weights2) + biases2, name="model")
    cost = -tf.reduce_sum(actual_classes * tf.log(model), name="cost")
    train_op1 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    init = tf.global_variables_initializer()

    # Run Model
    sess.run(init)
    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(actual_classes, 1), name="prediction")
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")
    for i in range(1, 30001):
        sess.run(
            train_op1,
            feed_dict={
                feature_data: training_inputs.values,
                actual_classes: training_outputs.values.reshape(len(training_outputs.values), 2)
            }
        )
        # Every 5000, we are going to print the current training accuracy of the model
        if i % 5000 == 0:
            print("start date", start, "iteration", i, sess.run(
                accuracy,
                feed_dict={
                    feature_data: training_inputs.values,
                    actual_classes: training_outputs.values.reshape(len(training_outputs.values), 2)
                }
            ))

    feed_dict = {
        feature_data: test_inputs.values,
        actual_classes: test_outputs.values.reshape(len(test_outputs.values), 2)
    }

    # Calculate the F1 Score and Accuracy with the training set
    f1_score, accuracy = tf_confusion_metrics(model, actual_classes, sess, feed_dict)
    print(start, f1_score, accuracy)

    saver = tf.train.Saver()

    test_dict = { # mongodb can't save numpy arrays. convert back with pickle.loads(x)
        'feature_data': Binary(pickle.dumps(test_inputs.values, protocol=2)),
        'actual_classes': Binary(pickle.dumps(test_outputs.values.reshape(len(test_outputs.values), 2), protocol=2))
    }

    train_dict = {
        'feature_data': Binary(pickle.dumps(training_inputs.values, protocol=2)),
        'actual_classes': Binary(pickle.dumps(training_outputs.values.reshape(len(training_outputs.values), 2), protocol=2))
    }

    return saver, sess, test_dict, train_dict, f1_score, accuracy


def load_model(date_id):
    """
    Load a tensor model from disk
    :param date_id: the id
    :return:
    """
    # MongoDB
    client = MongoClient('localhost', 27017)
    db = client['thesis']
    posts = db.posts

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('models/'+date_id+"/"+ date_id + ".meta")
        saver.restore(sess, tf.train.latest_checkpoint("models/" + date_id + "/"))
        graph = tf.get_default_graph()
        model = graph.get_tensor_by_name("model:0")
        actual_classes = graph.get_tensor_by_name("actual_classes:0")
        feature_data = graph.get_tensor_by_name("feature_data:0")

        post = posts.find_one({"_id": date_id})
        test_fd = pickle.loads(post['test_data']['feature_data'])
        test_ac = pickle.loads(post['test_data']['actual_classes'])

        feed_dict = {
            feature_data: test_fd,
            actual_classes: test_ac
        }

        print(tf_confusion_metrics(model, actual_classes, sess, feed_dict))
        print(tf_confusion_metrics(model, actual_classes, sess, feed_dict))
        print(tf_confusion_metrics(model, actual_classes, sess, feed_dict))