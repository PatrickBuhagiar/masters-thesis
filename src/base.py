import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, wait

# global variables
from stock_model import divide_into_training_testing
from toolbox import extract_index

start = pd.datetime(2013, 1, 1)
end = pd.datetime(2018, 1, 1)

# Concurrency stuff
pool = ThreadPoolExecutor(4)
futures = []


def prepare_data():
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


# load FTSE and prepare data


def run(learn_rate, n_nodes, training_inputs, training_outputs, test_inputs, test_outputs):
    feature_count = training_inputs.shape[1]
    label_count = training_outputs.shape[1]
    training_epochs = 3000

    cost_history = np.empty(shape=[1], dtype=float)
    X = tf.placeholder(tf.float32, [None, feature_count])
    Y = tf.placeholder(tf.float32, [None, label_count])
    initializer = tf.contrib.layers.xavier_initializer()
    h0 = tf.layers.dense(X, n_nodes, activation=tf.nn.relu, kernel_initializer=initializer)
    h0 = tf.nn.dropout(h0, 0.80)
    h1 = tf.layers.dense(h0, label_count, activation=None)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=h1)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)
    predicted = tf.nn.sigmoid(h1)
    correct_pred = tf.equal(tf.round(predicted), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    TP = tf.count_nonzero(tf.round(predicted) * Y)
    TN = tf.count_nonzero((tf.round(predicted) - 1) * (Y - 1))
    FP = tf.count_nonzero(tf.round(predicted) * (Y - 1))
    FN = tf.count_nonzero((tf.round(predicted) - 1) * Y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(training_epochs + 1):
            sess.run(optimizer, feed_dict={X: training_inputs, Y: training_outputs})
            loss, _, acc = sess.run([cost, optimizer, accuracy], feed_dict={X: training_inputs, Y: training_outputs})
            cost_history = np.append(cost_history, acc)

            # if step % 500 == 0:
            #     print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
        # test_predict_result = sess.run(tf.cast(tf.round(predicted), tf.int32), feed_dict={X: test_inputs})
        return sess.run([accuracy, TP, TN, FP, FN], feed_dict={X: test_inputs, Y: test_outputs})


def process_with_learning_rate(j, X, Y, Z, ZZ, training_inputs, training_outputs, test_inputs, test_outputs):
    learning_rate = Y[j]
    for i in range(0, len(X)):
        n_nodes = X[i]
        acc = 0.0
        f1 = 0.0
        for k in range(0, 20):
            accuracy, TP, TN, FP, FN = run(learning_rate, n_nodes, training_inputs, training_outputs, test_inputs,
                                           test_outputs)
            acc += accuracy
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 += (2 * precision * recall) / (precision + recall)
            print("learning rate", learning_rate, "n_nodes", n_nodes, "iter", k, "f1",
                  (2 * precision * recall) / (precision + recall), "accuracy", accuracy, TP, TN, FP, FN)
        acc = acc / 20.0
        f1 = f1 / 20.0
        print("learning rate", learning_rate, "n_nodes", n_nodes, "TOTAL", "f1",
              f1, "accuracy", acc)

        Z[i][j] = acc
        ZZ[i][j] = f1


if __name__ == '__main__':
    # test_outputs, test_inputs, training_outputs, training_inputs = prepare_data()
    X = np.arange(4, 27, 2)  # number of nodes
    Y = np.arange(0.001, 0.012, 0.002)  # learning rates
    # accuracies = np.ones([len(X), len(Y)])
    # f1s = np.ones([len(X), len(Y)])
    # for j in range(0, len(Y)):
    #     futures.append(
    #         pool.submit(process_with_learning_rate, j, X, Y, accuracies, f1s, training_inputs, training_outputs,
    #                     test_inputs,
    #                     test_outputs))
    #
    # wait(futures)
    # np.savetxt("final_accuracies.csv", accuracies, delimiter=",")
    # np.savetxt("final_f1s.csv", f1s, delimiter=",")

    accuracies = np.array(list(csv.reader(open("final_accuracies.csv"), delimiter=","))).astype("float")
    Y, X = np.meshgrid(Y, X)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Number of Hidden Layer Nodes")
    ax.set_zlabel("Accuracy")

    plt.title("3D plot of Number of Nodes VS Learning Rate VS Accuracy")
    surf = ax.plot_surface(Y, X, accuracies, cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0)
    plt.show()