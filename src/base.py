from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import pandas as pd
import tensorflow as tf

start = pd.datetime(2013, 1, 1)
end = pd.datetime(2018, 1, 1)

# Concurrency stuff
pool = ThreadPoolExecutor(20)
futures = []


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


def prepare_data():
    """
    Loads the data, preprocesses it and prepares it for the feed forward network
    :return: training and testing inputs and outputs
    """
    dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')
    ts_ftse = extract_index('../data/indices/^FTSE.csv', start, end, dateparse)
    ts_normalised = ts_ftse / max(ts_ftse)
    ts_log = np.log(ts_normalised / ts_normalised.shift())
    ts_log = ts_log.dropna()
    directions = pd.DataFrame()
    directions['FTSE'] = ts_log
    directions['UP'] = 0
    directions.ix[directions['FTSE'] >= 0, 'UP'] = 1
    directions['DOWN'] = 0
    directions.ix[directions['FTSE'] < 0, 'DOWN'] = 1
    data = pd.DataFrame(
        columns=['up', 'ftse_1', 'ftse_2', 'ftse_3', 'ftse_4', 'ftse_5']
    )
    for i in range(7, len(ts_log)):
        up = directions['UP'].ix[i]
        down = directions['DOWN'].ix[i]
        ftse_1 = ts_log.ix[i - 1]
        ftse_2 = ts_log.ix[i - 2]
        ftse_3 = ts_log.ix[i - 3]
        ftse_4 = ts_log.ix[i - 4]
        ftse_5 = ts_log.ix[i - 5]
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


def run(lr_rate, n_nodes, trn_inputs, trn_outputs, tst_inputs, tst_outputs):
    """
    Run the model with given parameters
    :param lr_rate: learning rate
    :param n_nodes: number of hidden nodes
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
    # Add drop out
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
            accuracy, t_p, t_n, f_p, f_n = run(learning_rate, n_nodes, trn_inputs, trn_outputs, tst_inputs,
                                               tst_outputs)
            acc += (t_p + t_n) / (t_p + t_n + f_p + f_n)
            precision = t_p / (t_p + f_p)
            recall = t_p / (t_p + f_n)
            f1 += 2 * ((precision * recall) / (precision + recall))
            print("learning rate", "%.5f" % learning_rate, "n_nodes", n_nodes, "iter", k, "f1",
                  (2 * precision * recall) / (precision + recall), "accuracy", (t_p + t_n) / (t_p + t_n + f_p + f_n),
                  t_p, t_n,
                  f_p, f_n)
        acc = acc / 20.0
        f1 = f1 / 20.0
        print("learning rate", "%.5f" % learning_rate, "n_nodes", n_nodes, "TOTAL", "f1",
              f1, "accuracy", acc)

        acc_matrix[i][j] = acc
        f1_matrix[i][j] = f1


if __name__ == '__main__':
    # get inputs
    test_outputs, test_inputs, training_outputs, training_inputs = prepare_data()
    X = np.arange(5, 11, 1)  # number of nodes
    Y = np.arange(0.0005, 0.0021, 0.0001)  # learning rates

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
    np.savetxt("base_accuracies_5-11_0005-0021.csv", accuracies, delimiter=",")
    np.savetxt("base_f1s_5-11_0005-0021.csv", f1s, delimiter=",")
