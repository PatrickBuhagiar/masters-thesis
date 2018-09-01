from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import pandas as pd
import tensorflow as tf

# Concurrency stuff
pool = ThreadPoolExecutor(20)
futures = []

# learning_rate = 0.0007
# n_nodes = 9


def extract_index(filename, start, end, date_parse, dropna=True):
    """
    Extracts the index from a csv file and filters base_out into a date range.

    :param  filename: The name of the csv file
    :param     start: The start date
    :param       end: the end date
    :param date_parse: the type of date parsing
    :param dropna: drop any nas

    :return: The indices as a time series
    """
    data = pd.read_csv(filename, parse_dates=['Date'], index_col='Date', date_parser=date_parse)
    # Fill missing dates and values
    all_days = pd.date_range(start, end, freq='D')
    data = data.reindex(all_days)
    ts = data['Close']
    if dropna:
        ts = ts.dropna()
    return ts


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


def prepare_data(ftse_data):
    directions = pd.DataFrame()
    directions['FTSE'] = ftse_data
    directions['UP'] = 0
    directions.ix[directions['FTSE'] >= 0, 'UP'] = 1
    directions['DOWN'] = 0
    directions.ix[directions['FTSE'] < 0, 'DOWN'] = 1
    data = pd.DataFrame(
        columns=['up', 'ftse_1', 'ftse_2', 'ftse_3', 'ftse_4', 'ftse_5']
    )
    for i in range(7, len(ftse_data)):
        up = directions['UP'].ix[i]
        down = directions['DOWN'].ix[i]
        ftse_1 = ftse_data.ix[i - 1]
        ftse_2 = ftse_data.ix[i - 2]
        ftse_3 = ftse_data.ix[i - 3]
        ftse_4 = ftse_data.ix[i - 4]
        ftse_5 = ftse_data.ix[i - 5]
        data = data.append(
            {
                'up': up,
                'ftse_1': ftse_1,
                'ftse_2': ftse_2,
                'ftse_3': ftse_3,
                'ftse_4': ftse_4,
                'ftse_5': ftse_5
            }, ignore_index=True
        )
    inputs = data[data.columns[1:]]
    outputs = data[data.columns[:1]]
    return divide_into_training_testing(inputs, outputs, len(data))


def load_data(start, end):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')
    ts_FTSE = extract_index('../data/indices/^FTSE.csv', start, end, dateparse)
    ts_normalised = ts_FTSE / max(ts_FTSE)
    ts_log = np.log(ts_normalised / ts_normalised.shift())
    ts_log = ts_log.dropna()
    return ts_log


# load FTSE and prepare data


def run(learn_rate, n_nodes, training_inputs, training_outputs, test_inputs, test_outputs):
    feature_count = training_inputs.shape[1]
    label_count = training_outputs.shape[1]
    training_epochs = 3000

    cost_history = np.empty(shape=[1], dtype=float)
    X = tf.placeholder(tf.float32, [None, feature_count], name="X")
    Y = tf.placeholder(tf.float32, [None, label_count], name="Y")
    initializer = tf.contrib.layers.xavier_initializer()
    h0 = tf.layers.dense(X, n_nodes, activation=tf.nn.relu, kernel_initializer=initializer)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h0 = tf.nn.dropout(h0, keep_prob)
    h1 = tf.layers.dense(h0, label_count, activation=None)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=h1, name="cross_entropy")
    cost = tf.reduce_mean(cross_entropy, name="cost")
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)
    predicted = tf.nn.sigmoid(h1, name="predicted")
    correct_pred = tf.equal(tf.round(predicted), Y)
    TP = tf.count_nonzero(tf.round(predicted) * Y)
    TN = tf.count_nonzero((tf.round(predicted) - 1) * (Y - 1))
    FP = tf.count_nonzero(tf.round(predicted) * (Y - 1))
    FN = tf.count_nonzero((tf.round(predicted) - 1) * Y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(training_epochs + 1):
        sess.run(optimizer, feed_dict={X: training_inputs, Y: training_outputs, keep_prob: 0.8})
        loss, _, acc = sess.run([cost, optimizer, accuracy],
                                feed_dict={X: training_inputs, Y: training_outputs, keep_prob: 0.8})
        cost_history = np.append(cost_history, acc)
    accuracy, TP, TN, FP, FN = sess.run([accuracy, TP, TN, FP, FN],
                                        feed_dict={X: test_inputs, Y: test_outputs, keep_prob: 1})
    saver = tf.train.Saver()
    return (TP + TN) / (TP + TN + FP + FN), TP, TN, FP, FN, saver, sess


# process for every date
def process(learning_rates, n_nodes, training_inputs, training_outputs, test_inputs, test_outputs, date):
    acc = 0.0
    f1 = 0.0
    lr = 0.0
    n_n = 0
    svr = None
    ses = None
    for learning_rate in learning_rates:
        for n_node in n_nodes:
            for k in range(0, 3):
                accuracy, TP, TN, FP, FN, saver, sess = run(learning_rate, n_node, training_inputs, training_outputs,
                                                            test_inputs,
                                                            test_outputs)
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1_score = (2 * precision * recall) / (precision + recall)
                accuracy = (TP + TN) / (TP + TN + FP + FN)
                if f1_score > f1:
                    if ses is not None:
                        ses.close()
                    f1 = f1_score
                    acc = accuracy
                    svr = saver
                    ses = sess
                    lr = learning_rate
                    n_n = n_node
                else:
                    sess.close()
                print("learning rate", "%.5f" % learning_rate, "n_nodes", n_node, "iter", k, "f1",
                      (2 * precision * recall) / (precision + recall), "accuracy", accuracy, TP, TN, FP, FN)

    print("Chosen Model for date", date, " is f1", f1, "accuracy", acc, "learning rate", "%.5f" % lr, "n_nodes", n_n)
    svr.save(ses, "h2_models2/" + date.date().__str__() + "/" + date.date().__str__())
    ses.close()


if __name__ == '__main__':
    start_years = np.arange(2002, 2006, 1)
    start_dates = []
    for year in start_years:
        start_dates.append(pd.datetime(year, 1, 1))
        start_dates.append(pd.datetime(year, 7, 1))

    for date in start_dates:
        ftse_data = load_data(date, date + pd.DateOffset(years=5))
        test_outputs, test_inputs, training_outputs, training_inputs = prepare_data(ftse_data)
        n_nodes = np.arange(6, 11, 1)  # number of nodes
        learning_rates = np.arange(0.0005, 0.0012, 0.0002)  # learning rates
        futures.append(
            pool.submit(process, learning_rates, n_nodes, training_inputs, training_outputs,
                        test_inputs,
                        test_outputs, date))

    wait(futures)
