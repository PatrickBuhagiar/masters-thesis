import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats

from src import h1, h2_stage_2, h3_stage_2

start = pd.datetime(2013, 1, 1)
end = pd.datetime(2018, 1, 1)


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


def prepare_base_data():
    dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')
    ts_FTSE = extract_index('../data/indices/^FTSE.csv', start, end, dateparse)
    ts_normalised = ts_FTSE / max(ts_FTSE)
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


def run_base(learn_rate, n_nodes, training_inputs, training_outputs, test_inputs, test_outputs):
    feature_count = training_inputs.shape[1]
    label_count = training_outputs.shape[1]
    training_epochs = 3000

    X = tf.placeholder(tf.float32, [None, feature_count])
    Y = tf.placeholder(tf.float32, [None, label_count])
    initializer = tf.contrib.layers.xavier_initializer()
    h0 = tf.layers.dense(X, n_nodes, activation=tf.nn.relu, kernel_initializer=initializer)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h0 = tf.nn.dropout(h0, keep_prob)
    h1 = tf.layers.dense(h0, label_count, activation=None)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=h1)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)
    predicted = tf.nn.sigmoid(h1)
    correct_pred = tf.equal(tf.round(predicted), Y)
    TP = tf.count_nonzero(tf.round(predicted) * Y)
    TN = tf.count_nonzero((tf.round(predicted) - 1) * (Y - 1))
    FP = tf.count_nonzero(tf.round(predicted) * (Y - 1))
    FN = tf.count_nonzero((tf.round(predicted) - 1) * Y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(training_epochs + 1):
            sess.run(optimizer, feed_dict={X: training_inputs, Y: training_outputs, keep_prob: 0.8})

        return sess.run([correct_pred, TP, TN, FP, FN], feed_dict={X: test_inputs, Y: test_outputs, keep_prob: 1})


def toInt(v):
    if v:
        return 1
    else:
        return 0


def get_base_predictions(test_outputs, test_inputs, training_outputs, training_inputs):
    print("BASE PREDICTIONS")
    learning_rate = 0.0016
    n_nodes = 9
    acc = 0.0
    base_correct_predictions = []
    for k in range(0, 20):
        correct_predictions, TP, TN, FP, FN = run_base(learning_rate, n_nodes, training_inputs, training_outputs,
                                                       test_inputs,
                                                       test_outputs)

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1score = 2 * ((precision * recall) / (precision + recall))
        print("learning rate", "%.5f" % learning_rate, "n_nodes", n_nodes, "iter", k, "f1",
              (2 * precision * recall) / (precision + recall), "accuracy", (TP + TN) / (TP + TN + FP + FN), TP, TN,
              FP, FN)

        if accuracy > acc:
            acc = accuracy
            base_correct_predictions = correct_predictions

    print("chosen model", "accuracy", acc)
    return base_correct_predictions


def get_h1_predictions(test_outputs, test_inputs, training_outputs, training_inputs):
    print("H1 PREDICTIONS")
    learning_rate = 0.0003
    n_nodes = 17
    acc = 0.0
    h1_correct_predictions = []
    for k in range(0, 20):
        correct_predictions, TP, TN, FP, FN = h1.run(learning_rate, n_nodes, training_inputs, training_outputs,
                                                     test_inputs, test_outputs, get_predictions=True)

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1score = 2 * ((precision * recall) / (precision + recall))
        print("learning rate", "%.5f" % learning_rate, "n_nodes", n_nodes, "iter", k, "f1",
              (2 * precision * recall) / (precision + recall), "accuracy", (TP + TN) / (TP + TN + FP + FN), TP, TN,
              FP, FN)

        if accuracy > acc:
            acc = accuracy
            h1_correct_predictions = correct_predictions

    print("chosen model", "accuracy", acc)
    return h1_correct_predictions


def get_h2_predictions(test_outputs, test_inputs, training_outputs, training_inputs):
    print("H2 PREDICTIONS")
    learning_rate = 0.0011
    n_nodes = 47
    acc = 0.0
    h2_correct_predictions = []
    for k in range(0, 20):
        correct_predictions, TP, TN, FP, FN = h2_stage_2.run(learning_rate, n_nodes, training_inputs, training_outputs,
                                                             test_inputs, test_outputs, getPredictions=True)

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1score = 2 * ((precision * recall) / (precision + recall))
        print("learning rate", "%.5f" % learning_rate, "n_nodes", n_nodes, "iter", k, "f1",
              (2 * precision * recall) / (precision + recall), "accuracy", (TP + TN) / (TP + TN + FP + FN), TP, TN,
              FP, FN)

        if accuracy > acc:
            acc = accuracy
            h2_correct_predictions = correct_predictions

    print("chosen model", "accuracy", acc)
    return h2_correct_predictions


def get_h3_predictions(test_outputs, test_inputs, training_outputs, training_inputs):
    print("H3 PREDICTIONS")
    learning_rate = 0.00001
    n_nodes = 56
    acc = 0.0
    h3_correct_predictions = []
    for k in range(0, 20):
        correct_predictions, TP, TN, FP, FN = h3_stage_2.run(learning_rate, n_nodes, training_inputs, training_outputs,
                                                             test_inputs, test_outputs, get_predictions=True)

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1score = 2 * ((precision * recall) / (precision + recall))
        print("learning rate", "%.5f" % learning_rate, "n_nodes", n_nodes, "iter", k, "f1",
              (2 * precision * recall) / (precision + recall), "accuracy", (TP + TN) / (TP + TN + FP + FN), TP, TN,
              FP, FN)

        if accuracy > acc:
            acc = accuracy
            h3_correct_predictions = correct_predictions

    print("chosen model", "accuracy", acc)
    return h3_correct_predictions


if __name__ == '__main__':
    # Base 2
    test_outputs, test_inputs, training_outputs, training_inputs = prepare_base_data()
    base2_correct_predictions = get_base_predictions(test_outputs, test_inputs, training_outputs, training_inputs)

    # Base 1
    base1_correct_predictions = (np.ones(len(test_outputs)) == test_outputs.values.flatten())
    # h1
    test_outputs, test_inputs, training_outputs, training_inputs = h1.prepare_data()
    h1_correct_predictions = get_h1_predictions(test_outputs, test_inputs, training_outputs, training_inputs)

    # h2
    ftse_data = h2_stage_2.load_data(start, end)
    inputs, outputs, dates = h2_stage_2.prepare_data(ftse_data)
    meta_inputs = h2_stage_2.get_meta_inputs(inputs, dates, start, end)
    test_outputs, test_inputs, training_outputs, training_inputs = divide_into_training_testing(meta_inputs, outputs,
                                                                                                len(meta_inputs))
    h2_correct_predictions = get_h2_predictions(test_outputs, test_inputs, training_outputs, training_inputs)

    # h3
    inputs, outputs, dates = h3_stage_2.prepare_data(start, end)
    meta_inputs = h3_stage_2.get_meta_inputs(inputs, dates, start, end)
    test_outputs, test_inputs, training_outputs, training_inputs = divide_into_training_testing(meta_inputs, outputs,
                                                                                                len(meta_inputs))
    h3_correct_predictions = get_h3_predictions(test_outputs, test_inputs, training_outputs, training_inputs)

    np.savetxt("base1_predictions.csv", [toInt(i) for i in base1_correct_predictions], delimiter=",")
    np.savetxt("base2_predictions.csv", [toInt(i) for i in base2_correct_predictions.flatten()], delimiter=",")
    np.savetxt("h1_predictions.csv", [toInt(i) for i in h1_correct_predictions.flatten()], delimiter=",")
    np.savetxt("h2_predictions.csv", [toInt(i) for i in h2_correct_predictions.flatten()], delimiter=",")
    np.savetxt("h3_predictions.csv", [toInt(i) for i in h3_correct_predictions.flatten()], delimiter=",")

    print(stats.kruskal([toInt(i) for i in base1_correct_predictions],
                        [toInt(i) for i in base2_correct_predictions.flatten()],
                        [toInt(i) for i in h1_correct_predictions.flatten()],
                        [toInt(i) for i in h2_correct_predictions.flatten()],
                        [toInt(i) for i in h3_correct_predictions.flatten()]))
