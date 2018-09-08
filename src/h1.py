from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import pandas as pd
import tensorflow as tf

# global variables
start = pd.datetime(2013, 1, 1)
end = pd.datetime(2018, 1, 1)

# Concurrency stuff
pool = ThreadPoolExecutor(20)
futures = []


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
    trn_inputs = inputs[:training_set_size]
    trn_outputs = outputs[:training_set_size]
    tst_inputs = inputs[training_set_size:]
    tst_outputs = outputs[training_set_size:]
    return tst_outputs, tst_inputs, trn_outputs, trn_inputs


def extract_index(filename, start_date, end_date, date_parse, dropna=True):
    """
    Extracts the index from a csv file and filters base_out into a date range.
    :param  filename: The name of the csv file
    :param     start_date: The start date
    :param       end_date: the end date
    :param date_parse: the type of date parsing
    :param dropna: drop any nas
    :return: The indices as a time series
    """
    data = pd.read_csv(filename, parse_dates=['Date'], index_col='Date', date_parser=date_parse)
    # Fill missing dates and values
    all_days = pd.date_range(start_date, end_date, freq='D')
    data = data.reindex(all_days)
    ts = data['Close']
    if dropna:
        ts = ts.dropna()
    return ts


def prepare_data():
    """
    Loads the data, preprocesses it and prepares it for the feed forward network
    :return: training and testing inputs and outputs
    """
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    dateparse2 = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y')
    dateparse3 = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')

    # Prepare FTSE
    ftse_data = extract_index('../data/indices/^FTSE.csv', start, end, dateparse3)
    ftse_normalised = ftse_data / max(ftse_data)
    ftse_log = np.log(ftse_normalised / ftse_normalised.shift())
    ftse_log = ftse_log.dropna()

    # Prepare other stocks
    cac_data = extract_index('../data/indices/^FCHI.csv', start, end, dateparse)
    dax_data = extract_index('../data/indices/^GDAXI.csv', start, end, dateparse)
    sp500_data = extract_index('../data/indices/^GSPC.csv', start, end, dateparse2)
    n225_data = extract_index('../data/indices/^N225.csv', start, end, dateparse)
    stoxx_data = extract_index('../data/indices/^STOXX50E.csv', start, end, dateparse2)
    hkse_data = extract_index('../data/indices/^HSI.csv', start, end, dateparse)

    cac_normalised = cac_data / max(cac_data)
    dax_normalised = dax_data / max(dax_data)
    sp500_normalised = sp500_data / max(sp500_data)
    n225_normalised = n225_data / max(n225_data)
    stoxx_normalised = stoxx_data / max(stoxx_data)
    hkse_normalised = hkse_data / max(hkse_data)

    cac_log = np.log(cac_normalised / cac_normalised.shift())
    dax_log = np.log(dax_normalised / dax_normalised.shift())
    sp500_log = np.log(sp500_normalised / sp500_normalised.shift())
    n225_log = np.log(n225_normalised / n225_normalised.shift())
    stoxx_log = np.log(stoxx_normalised / stoxx_normalised.shift())
    hkse_log = np.log(hkse_normalised / hkse_normalised.shift())

    cac_log = cac_log.dropna()
    dax_log = dax_log.dropna()
    sp500_log = sp500_log.dropna()
    stoxx_log = stoxx_log.dropna()
    hkse_log = hkse_log.dropna()
    n225_log = n225_log.dropna()

    # Prepare directions
    directions = pd.DataFrame()
    directions['FTSE'] = ftse_log
    directions['UP'] = 0
    directions.ix[directions['FTSE'] >= 0, 'UP'] = 1
    directions['DOWN'] = 0
    directions.ix[directions['FTSE'] < 0, 'DOWN'] = 1

    data = pd.DataFrame(
        columns=['up', 'ftse_1', 'ftse_2', 'ftse_3', 'cac_1', 'cac_2', 'cac_3', 'dax_1', 'dax_2', 'dax_3', 'stoxx_1',
                 'stoxx_2', 'stoxx_3', 'sp500_1', 'sp500_2', 'sp500_3', 'hkse_0', 'hkse_1', 'hkse_2', 'n225_0',
                 'n225_1', 'n225_2']
    )
    for i in range(7, len(ftse_log)):
        up = directions['UP'].ix[i]

        date_0 = ftse_log.keys()[i]
        date_1 = ftse_log.keys()[i - 1]
        date_2 = ftse_log.keys()[i - 2]
        date_3 = ftse_log.keys()[i - 3]
        date_4 = ftse_log.keys()[i - 4]
        date_5 = ftse_log.keys()[i - 5]
        date_6 = ftse_log.keys()[i - 6]

        ftse_1 = ftse_log.ix[i - 1]
        ftse_2 = ftse_log.ix[i - 2]
        ftse_3 = ftse_log.ix[i - 3]

        cac_1 = 0
        cac_2 = 0
        cac_3 = 0
        if cac_log.keys().contains(date_3):
            cac_3 = cac_log.ix[date_3]
        elif cac_log.keys().contains(date_4):
            cac_3 = cac_log.ix[date_4]
        elif cac_log.keys().contains(date_5):
            cac_3 = cac_log.ix[date_5]
        else:
            cac_3 = cac_log.ix[date_6]

        if cac_log.keys().contains(date_2):
            cac_2 = cac_log.ix[date_2]
        else:
            cac_2 = cac_3

        if cac_log.keys().contains(date_1):
            cac_1 = cac_log.ix[date_1]
        else:
            cac_1 = cac_2

        dax_1 = 0
        dax_2 = 0
        dax_3 = 0
        if dax_log.keys().contains(date_3):
            dax_3 = dax_log.ix[date_3]
        elif dax_log.keys().contains(date_4):
            dax_3 = dax_log.ix[date_4]
        elif dax_log.keys().contains(date_5):
            dax_3 = dax_log.ix[date_5]
        else:
            dax_3 = dax_log.ix[date_6]

        if dax_log.keys().contains(date_2):
            dax_2 = dax_log.ix[date_2]
        else:
            dax_2 = dax_3

        if dax_log.keys().contains(date_1):
            dax_1 = dax_log.ix[date_1]
        else:
            dax_1 = dax_2

        stoxx_1 = 0
        stoxx_2 = 0
        stoxx_3 = 0
        if stoxx_log.keys().contains(date_3):
            stoxx_3 = stoxx_log.ix[date_3]
        elif stoxx_log.keys().contains(date_4):
            stoxx_3 = stoxx_log.ix[date_4]
        elif stoxx_log.keys().contains(date_5):
            stoxx_3 = stoxx_log.ix[date_5]
        else:
            stoxx_3 = stoxx_log.ix[date_6]

        if stoxx_log.keys().contains(date_2):
            stoxx_2 = stoxx_log.ix[date_2]
        else:
            stoxx_2 = stoxx_3

        if stoxx_log.keys().contains(date_1):
            stoxx_1 = stoxx_log.ix[date_1]
        else:
            stoxx_1 = stoxx_2

        sp500_1 = 0
        sp500_2 = 0
        sp500_3 = 0
        if sp500_log.keys().contains(date_3):
            sp500_3 = sp500_log.ix[date_3]
        elif sp500_log.keys().contains(date_4):
            sp500_3 = sp500_log.ix[date_4]
        elif sp500_log.keys().contains(date_5):
            sp500_3 = sp500_log.ix[date_5]
        else:
            sp500_3 = sp500_log.ix[date_6]

        if sp500_log.keys().contains(date_2):
            sp500_2 = sp500_log.ix[date_2]
        else:
            sp500_2 = sp500_3

        if sp500_log.keys().contains(date_1):
            sp500_1 = sp500_log.ix[date_1]
        else:
            sp500_1 = sp500_2

        hkse_0 = 0
        hkse_1 = 0
        hkse_2 = 0
        if hkse_log.keys().contains(date_2):
            hkse_2 = hkse_log.ix[date_2]
        elif hkse_log.keys().contains(date_3):
            hkse_2 = hkse_log.ix[date_3]
        elif hkse_log.keys().contains(date_4):
            hkse_2 = hkse_log.ix[date_4]
        else:
            hkse_2 = hkse_log.ix[date_5]

        if hkse_log.keys().contains(date_1):
            hkse_1 = hkse_log.ix[date_1]
        else:
            hkse_1 = hkse_2

        if hkse_log.keys().contains(date_0):
            hkse_0 = hkse_log.ix[date_0]
        else:
            hkse_0 = hkse_1

        n225_0 = 0
        n225_1 = 0
        n225_2 = 0
        if n225_log.keys().contains(date_2):
            n225_2 = n225_log.ix[date_2]
        elif n225_log.keys().contains(date_3):
            n225_2 = n225_log.ix[date_3]
        elif n225_log.keys().contains(date_4):
            n225_2 = n225_log.ix[date_4]
        elif n225_log.keys().contains(date_5):
            n225_2 = n225_log.ix[date_5]
        else:
            n225_2 = n225_log.ix[date_6]

        if n225_log.keys().contains(date_1):
            n225_1 = n225_log.ix[date_1]
        else:
            n225_1 = n225_2

        if n225_log.keys().contains(date_0):
            n225_0 = n225_log.ix[date_0]
        else:
            n225_0 = n225_1
        data = data.append(
            {
                'up': up,
                'ftse_1': ftse_1,
                'ftse_2': ftse_2,
                'ftse_3': ftse_3,
                'cac_1': cac_1,
                'cac_2': cac_2,
                'cac_3': cac_3,
                'dax_1': dax_1,
                'dax_2': dax_2,
                'dax_3': dax_3,
                'stoxx_1': stoxx_1,
                'stoxx_2': stoxx_2,
                'stoxx_3': stoxx_3,
                'sp500_1': sp500_1,
                'sp500_2': sp500_2,
                'sp500_3': sp500_3,
                'hkse_0': hkse_0,
                'hkse_1': hkse_1,
                'hkse_2': hkse_2,
                'n225_0': n225_0,
                'n225_1': n225_1,
                'n225_2': n225_2
            }, ignore_index=True
        )
    inputs = data[data.columns[1:]]
    outputs = data[data.columns[:1]]
    return divide_into_training_testing(inputs, outputs, len(data))


def run(lr_rate, n_nodes, trn_inputs, trn_outputs, tst_inputs, tst_outputs, get_predictions=False):
    """
      Run the model with given parameters
      :param get_predictions: If true, return predictions instead of just accuracy
      :param lr_rate: learning rate
      :param n_nodes: nmber of hidden nodes
      :param trn_inputs: training inputs
      :param trn_outputs: training outputs
      :param tst_inputs: testing inputs
      :param tst_outputs: testing outputs
      :return: accuracy, true positives, true negatives, false positives and false negatives
      """
    feature_count = trn_inputs.shape[1]
    label_count = trn_outputs.shape[1]
    training_epochs = 3000

    cost_history = np.empty(shape=[1], dtype=float)
    X = tf.placeholder(tf.float32, [None, feature_count])
    Y = tf.placeholder(tf.float32, [None, label_count])
    initializer = tf.contrib.layers.xavier_initializer()
    h0 = tf.layers.dense(X, n_nodes, activation=tf.nn.relu, kernel_initializer=initializer)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h0 = tf.nn.dropout(h0, keep_prob)
    h1 = tf.layers.dense(h0, label_count, activation=None)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=h1)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_rate).minimize(cost)
    predicted = tf.nn.sigmoid(h1)
    correct_pred = tf.equal(tf.round(predicted), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    t_p = tf.count_nonzero(tf.round(predicted) * Y)
    t_n = tf.count_nonzero((tf.round(predicted) - 1) * (Y - 1))
    f_p = tf.count_nonzero(tf.round(predicted) * (Y - 1))
    f_n = tf.count_nonzero((tf.round(predicted) - 1) * Y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(training_epochs + 1):
            sess.run(optimizer, feed_dict={X: trn_inputs, Y: trn_outputs, keep_prob: 0.8})
            loss, _, acc = sess.run([cost, optimizer, accuracy],
                                    feed_dict={X: trn_inputs, Y: trn_outputs, keep_prob: 0.8})
            cost_history = np.append(cost_history, acc)
        if get_predictions:
            return sess.run([correct_pred, t_p, t_n, f_p, f_n], feed_dict={X: tst_inputs, Y: tst_outputs, keep_prob: 1})
        else:
            return sess.run([accuracy, t_p, t_n, f_p, f_n], feed_dict={X: tst_inputs, Y: tst_outputs, keep_prob: 1})


def process_with_learning_rate(j, hidden_nodes_range, lr_rates, acc_matrix, f1_matrix, trn_inputs, trn_outputs,
                               tst_inputs, tst_outputs):
    """
    process with learning rate. this calls the run method
    :param j: for loop index
    :param hidden_nodes_range: range of number of hidden nodes to test with
    :param lr_rates: range of learning rates to test with
    :param acc_matrix: where final accuracy results are stored
    :param f1_matrix: where final f1 scores results are stored
    :param trn_inputs: training inputs
    :param trn_outputs: training outputs
    :param tst_inputs: testing inputs
    :param tst_outputs: testing outputs
    :return:
    """
    learning_rate = lr_rates[j]
    for i in range(0, len(hidden_nodes_range)):
        n_nodes = hidden_nodes_range[i]
        acc = 0.0
        f1 = 0.0
        for k in range(0, 20):
            accuracy, TP, TN, FP, FN = run(learning_rate, n_nodes, trn_inputs, trn_outputs, tst_inputs,
                                           tst_outputs)
            acc += (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 += 2 * ((precision * recall) / (precision + recall))
            print("learning rate", "%.5f" % learning_rate, "n_nodes", n_nodes, "iter", k, "f1",
                  (2 * precision * recall) / (precision + recall), "accuracy", (TP + TN) / (TP + TN + FP + FN), TP, TN,
                  FP, FN)
        acc = acc / 20.0
        f1 = f1 / 20.0
        print("learning rate", "%.5f" % learning_rate, "n_nodes", n_nodes, "TOTAL", "f1",
              f1, "accuracy", acc)

        acc_matrix[i][j] = acc
        f1_matrix[i][j] = f1


if __name__ == '__main__':
    # get inputs
    test_outputs, test_inputs, training_outputs, training_inputs = prepare_data()
    X = np.arange(10, 21, 1)  # number of nodes
    Y = np.arange(0.0001, 0.0009, 0.0001)  # learning rates

    # result matrices
    accuracies = np.ones([len(X), len(Y)])
    f1s = np.ones([len(X), len(Y)])

    # iterate over different combinations of parameters
    for j in range(0, len(Y)):
        futures.append(
            pool.submit(process_with_learning_rate, j, X, Y, accuracies, f1s, training_inputs, training_outputs,
                        test_inputs,
                        test_outputs))

    wait(futures)
    # store results
    np.savetxt("other_stocks_accuracies_10-21_0001-0009.csv", accuracies, delimiter=",")
    np.savetxt("other_stocks_f1s_10-21_0001-0009.csv", f1s, delimiter=",")
