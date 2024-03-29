from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import pandas as pd
import tensorflow as tf

# Concurrency stuff
pool = ThreadPoolExecutor(16)
futures = []


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


def prepare_data(ftse, cac, dax, sp500, stoxx, hkse, n225):
    directions = pd.DataFrame()
    directions['FTSE'] = ftse
    directions['UP'] = 0
    directions.ix[directions['FTSE'] >= 0, 'UP'] = 1
    directions['DOWN'] = 0
    directions.ix[directions['FTSE'] < 0, 'DOWN'] = 1

    data = pd.DataFrame(
        columns=['up', 'ftse_1', 'ftse_2', 'ftse_3', 'cac_1', 'cac_2', 'cac_3', 'dax_1', 'dax_2', 'dax_3', 'stoxx_1',
                 'stoxx_2', 'stoxx_3', 'sp500_1', 'sp500_2', 'sp500_3', 'hkse_0', 'hkse_1', 'hkse_2', 'n225_0',
                 'n225_1', 'n225_2']
    )
    for i in range(7, len(ftse)):
        up = directions['UP'].ix[i]

        date_0 = ftse.keys()[i]
        date_1 = ftse.keys()[i - 1]
        date_2 = ftse.keys()[i - 2]
        date_3 = ftse.keys()[i - 3]
        date_4 = ftse.keys()[i - 4]
        date_5 = ftse.keys()[i - 5]
        date_6 = ftse.keys()[i - 6]
        date_7 = ftse.keys()[i - 7]

        ftse_1 = ftse.ix[i - 1]
        ftse_2 = ftse.ix[i - 2]
        ftse_3 = ftse.ix[i - 3]

        cac_1 = 0
        cac_2 = 0
        cac_3 = 0
        if cac.keys().contains(date_3):
            cac_3 = cac.ix[date_3]
        elif cac.keys().contains(date_4):
            cac_3 = cac.ix[date_4]
        elif cac.keys().contains(date_5):
            cac_3 = cac.ix[date_5]
        else:
            cac_3 = cac.ix[date_6]

        if cac.keys().contains(date_2):
            cac_2 = cac.ix[date_2]
        else:
            cac_2 = cac_3

        if cac.keys().contains(date_1):
            cac_1 = cac.ix[date_1]
        else:
            cac_1 = cac_2

        dax_1 = 0
        dax_2 = 0
        dax_3 = 0
        if dax.keys().contains(date_3):
            dax_3 = dax.ix[date_3]
        elif dax.keys().contains(date_4):
            dax_3 = dax.ix[date_4]
        elif dax.keys().contains(date_5):
            dax_3 = dax.ix[date_5]
        else:
            dax_3 = dax.ix[date_6]

        if dax.keys().contains(date_2):
            dax_2 = dax.ix[date_2]
        else:
            dax_2 = dax_3

        if dax.keys().contains(date_1):
            dax_1 = dax.ix[date_1]
        else:
            dax_1 = dax_2

        stoxx_1 = 0
        stoxx_2 = 0
        stoxx_3 = 0
        if stoxx.keys().contains(date_3):
            stoxx_3 = stoxx.ix[date_3]
        elif stoxx.keys().contains(date_4):
            stoxx_3 = stoxx.ix[date_4]
        elif stoxx.keys().contains(date_5):
            stoxx_3 = stoxx.ix[date_5]
        else:
            stoxx_3 = stoxx.ix[date_6]

        if stoxx.keys().contains(date_2):
            stoxx_2 = stoxx.ix[date_2]
        else:
            stoxx_2 = stoxx_3

        if stoxx.keys().contains(date_1):
            stoxx_1 = stoxx.ix[date_1]
        else:
            stoxx_1 = stoxx_2

        sp500_1 = 0
        sp500_2 = 0
        sp500_3 = 0
        if sp500.keys().contains(date_3):
            sp500_3 = sp500.ix[date_3]
        elif sp500.keys().contains(date_4):
            sp500_3 = sp500.ix[date_4]
        elif sp500.keys().contains(date_5):
            sp500_3 = sp500.ix[date_5]
        elif sp500.keys().contains(date_6):
            sp500_3 = sp500.ix[date_6]
        else:
            sp500_3 = sp500.ix[date_7]

        if sp500.keys().contains(date_2):
            sp500_2 = sp500.ix[date_2]
        else:
            sp500_2 = sp500_3

        if sp500.keys().contains(date_1):
            sp500_1 = sp500.ix[date_1]
        else:
            sp500_1 = sp500_2

        hkse_0 = 0
        hkse_1 = 0
        hkse_2 = 0
        if hkse.keys().contains(date_2):
            hkse_2 = hkse.ix[date_2]
        elif hkse.keys().contains(date_3):
            hkse_2 = hkse.ix[date_3]
        elif hkse.keys().contains(date_4):
            hkse_2 = hkse.ix[date_4]
        else:
            hkse_2 = hkse.ix[date_5]

        if hkse.keys().contains(date_1):
            hkse_1 = hkse.ix[date_1]
        else:
            hkse_1 = hkse_2

        if hkse.keys().contains(date_0):
            hkse_0 = hkse.ix[date_0]
        else:
            hkse_0 = hkse_1

        n225_0 = 0
        n225_1 = 0
        n225_2 = 0
        if n225.keys().contains(date_2):
            n225_2 = n225.ix[date_2]
        elif n225.keys().contains(date_3):
            n225_2 = n225.ix[date_3]
        elif n225.keys().contains(date_4):
            n225_2 = n225.ix[date_4]
        elif n225.keys().contains(date_5):
            n225_2 = n225.ix[date_5]
        else:
            n225_2 = n225.ix[date_6]

        if n225.keys().contains(date_1):
            n225_1 = n225.ix[date_1]
        else:
            n225_1 = n225_2

        if n225.keys().contains(date_0):
            n225_0 = n225.ix[date_0]
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


def load_data(start, end):
    """
    load the data
    :param start: start date
    :param end: end date
    :return: date
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

    return ftse_log, cac_log, dax_log, sp500_log, stoxx_log, hkse_log, n225_log


def run(learn_rate, nodes, trn_inputs, trn_outputs, tst_inputs, tst_outputs, model_date):
    """
    Run model with given parameters
    :param learn_rate: learning rate
    :param nodes: number of hidden nodes
    :param trn_inputs: training inputs
    :param trn_outputs: training outputs
    :param tst_inputs: testing inputs
    :param tst_outputs: testing output
    :param model_date: the date
    :return:
    """
    tf.reset_default_graph()

    feature_count = trn_inputs.shape[1]
    label_count = trn_outputs.shape[1]
    training_epochs = 3000

    cost_history = np.empty(shape=[1], dtype=float)
    date_name = model_date.year.__str__() + model_date.month.__str__()

    X = tf.placeholder(tf.float32, [None, feature_count], name="X_" + date_name)
    Y = tf.placeholder(tf.float32, [None, label_count], name="Y_" + date_name)
    initializer = tf.contrib.layers.xavier_initializer()
    h0 = tf.layers.dense(X, nodes, activation=tf.nn.relu, kernel_initializer=initializer,
                         name="h0_" + date_name)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob_' + date_name)
    h0 = tf.nn.dropout(h0, keep_prob)
    h1 = tf.layers.dense(h0, label_count, activation=None, name="h1_" + date_name)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=h1,
                                                            name="cross_entropy_" + date_name)
    cost = tf.reduce_mean(cross_entropy, name="cost_" + date_name)
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)
    predicted = tf.nn.sigmoid(h1, name="predicted_" + date_name)
    correct_pred = tf.equal(tf.round(predicted), Y, name="correct_pred_" + date_name)
    TP = tf.count_nonzero(tf.round(predicted) * Y, name="TP_" + date_name)
    TN = tf.count_nonzero((tf.round(predicted) - 1) * (Y - 1), name="TN_" + date_name)
    FP = tf.count_nonzero(tf.round(predicted) * (Y - 1), name="FP_" + date_name)
    FN = tf.count_nonzero((tf.round(predicted) - 1) * Y, name="FN_" + date_name)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),
                              name="accuracy_" + date_name)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(training_epochs + 1):
        sess.run(optimizer, feed_dict={X: trn_inputs, Y: trn_outputs, keep_prob: 0.8})
        loss, _, acc = sess.run([cost, optimizer, accuracy],
                                feed_dict={X: trn_inputs, Y: trn_outputs, keep_prob: 0.8})
        cost_history = np.append(cost_history, acc)
    accuracy, TP, TN, FP, FN = sess.run([accuracy, TP, TN, FP, FN],
                                        feed_dict={X: tst_inputs, Y: tst_outputs, keep_prob: 1})
    saver = tf.train.Saver()
    return (TP + TN) / (TP + TN + FP + FN), TP, TN, FP, FN, saver, sess


def process(lr_rates, nodes, trn_inputs, trn_outputs, tst_inputs, tst_outputs, model_date):
    """
    Run models for combinations of parameters
    :param lr_rates: range of learning rates
    :param nodes: range of number of hidden nodes
    :param trn_inputs: training inputs
    :param trn_outputs: training outputs
    :param tst_inputs: testing inputs
    :param tst_outputs: testing outputs
    :param model_date: model date
    :return:
    """
    acc = 0.0
    f1 = 0.0
    lr = 0.0
    n_n = 0
    svr = None
    ses = None
    for learning_rate in lr_rates:
        for n_node in nodes:
            for k in range(0, 20):
                accuracy, TP, TN, FP, FN, saver, sess = run(learning_rate, n_node, trn_inputs, trn_outputs,
                                                            tst_inputs,
                                                            tst_outputs, model_date)
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1_score = (2 * precision * recall) / (precision + recall)
                accuracy = (TP + TN) / (TP + TN + FP + FN)
                if accuracy > acc:
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
                print("date", model_date, "learning rate", "%.5f" % learning_rate, "n_nodes", n_node, "iter", k, "f1",
                      (2 * precision * recall) / (precision + recall), "accuracy", accuracy, TP, TN, FP, FN)

    print("Chosen Model for date", model_date, " is f1", f1, "accuracy", acc, "learning rate", "%.5f" % lr, "n_nodes", n_n)
    svr.save(ses, "h3_models/" + model_date.date().__str__() + "/" + model_date.date().__str__())
    ses.close()


if __name__ == '__main__':
    start_years = np.arange(2000, 2008, 1)
    start_dates = []
    for year in start_years:
        start_dates.append(pd.datetime(year, 1, 1))
        start_dates.append(pd.datetime(year, 7, 1))

    lr = [0.0008, 0.0004, 0.0009, 0.0017, 0.0007, 0.0013, 0.0001, 0.0003,
          0.0003, 0.0002, 0.0003, 0.0005, 0.0005, 0.0001, 0.0009, 0.0015]

    nn = [12, 14, 16, 19, 17, 15, 23, 19,
          25, 12, 15, 17, 19, 17, 21, 23]

    for i in range(0, len(start_dates)):
        date = start_dates[i]
        ftse_log, cac_log, dax_log, sp500_log, stoxx_log, hkse_log, n225_log = load_data(date,
                                                                                         date + pd.DateOffset(years=5))
        test_outputs, test_inputs, training_outputs, training_inputs = prepare_data(ftse_log, cac_log, dax_log,
                                                                                    sp500_log, stoxx_log, hkse_log,
                                                                                    n225_log)
        n_nodes = [nn[i]]  # np.arange(8, 15, 1)  # number of nodes
        learning_rates = [lr[i]]  # np.arange(0.0003, 0.001, 0.0001)  # learning rates

        futures.append(
            pool.submit(process, learning_rates, n_nodes, training_inputs, training_outputs,
                        test_inputs,
                        test_outputs, date))

    wait(futures)
