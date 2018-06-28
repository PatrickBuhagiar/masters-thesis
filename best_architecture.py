import numpy as np
import pandas as pd
import tensorflow as tf

from toolbox import load_indices


def tf_confusion_metrics(model, actual_classes, session, feed_dict):
    # model is a 2 x 1456 matrix, which are the predictions made by the model.
    # argmax returns the index with the highest value, i.e. which did it predict correctly
    predictions = tf.argmax(model, 1)
    actuals = tf.argmax(actual_classes, 1)

    # returns a tensor with the same shape and type as actuals with all elements set to 1
    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    tp_op = tf.reduce_sum(  # computes the sum of elements across a dimension
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    tn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    fp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    fn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    tp, tn, fp, fn = \
        session.run(
            [tp_op, tn_op, fp_op, fn_op],
            feed_dict
        )

    tpr = float(tp) / (float(tp) + float(fn))

    accuracy = (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn))

    recall = tpr
    if recall != 0:
        precision = float(tp) / (float(tp) + float(fp))
        f1_score = (2 * (precision * recall) / (precision + recall))
    else:
        f1_score = 0
    return f1_score, accuracy


def run_model(n_hidden_nodes):
    # Load Data
    start = pd.datetime(2012, 2, 2)
    end = pd.datetime(2017, 2, 2)
    indices = load_indices(start, end)
    df = pd.DataFrame(index=indices['FTSE'].index)
    df['CAC'] = indices['CAC']
    df['DAX'] = indices['DAX']
    df['HKSE'] = indices['HKSE']
    df['NIKKEI'] = indices['NIKKEI']
    df['S&P500'] = indices['S&P500']
    df['STOXX'] = indices['STOXX']
    df['FTSE'] = indices['FTSE']
    df['CAC_scaled'] = df['CAC'] / max(df['CAC'])
    df['DAX_scaled'] = df['DAX'] / max(df['DAX'])
    df['HKSE_scaled'] = df['HKSE'] / max(df['HKSE'])
    df['NIKKEI_scaled'] = df['NIKKEI'] / max(df['NIKKEI'])
    df['S&P500_scaled'] = df['S&P500'] / max(df['S&P500'])
    df['STOXX_scaled'] = df['STOXX'] / max(df['STOXX'])
    df['FTSE_scaled'] = df['FTSE'] / max(df['FTSE'])
    log_return_data = pd.DataFrame()
    log_return_data['CAC_log_return'] = np.log(df['CAC'] / df['CAC'].shift())
    log_return_data['DAX_log_return'] = np.log(df['DAX'] / df['DAX'].shift())
    log_return_data['HKSE_log_return'] = np.log(df['HKSE'] / df['HKSE'].shift())
    log_return_data['NIKKEI_log_return'] = np.log(df['NIKKEI'] / df['NIKKEI'].shift())
    log_return_data['S&P500_log_return'] = np.log(df['S&P500'] / df['S&P500'].shift())
    log_return_data['STOXX_log_return'] = np.log(df['STOXX'] / df['STOXX'].shift())
    log_return_data['FTSE_log_return'] = np.log(df['FTSE'] / df['FTSE'].shift())
    log_return_data['ftse_log_return_positive'] = 0
    log_return_data.ix[log_return_data['FTSE_log_return'] >= 0, 'ftse_log_return_positive'] = 1
    log_return_data['ftse_log_return_negative'] = 0
    log_return_data.ix[log_return_data['FTSE_log_return'] < 0, 'ftse_log_return_negative'] = 1
    training_test_data = pd.DataFrame(
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
    for i in range(7, len(log_return_data)):
        ftse_log_return_positive = log_return_data['ftse_log_return_positive'].ix[i]
        ftse_log_return_negative = log_return_data['ftse_log_return_negative'].ix[i]
        ftse_log_return_1 = log_return_data['FTSE_log_return'].ix[i - 1]
        ftse_log_return_2 = log_return_data['FTSE_log_return'].ix[i - 2]
        ftse_log_return_3 = log_return_data['FTSE_log_return'].ix[i - 3]
        cac_log_return_1 = log_return_data['CAC_log_return'].ix[i - 1]
        cac_log_return_2 = log_return_data['CAC_log_return'].ix[i - 2]
        cac_log_return_3 = log_return_data['CAC_log_return'].ix[i - 3]
        dax_log_return_1 = log_return_data['DAX_log_return'].ix[i - 1]
        dax_log_return_2 = log_return_data['DAX_log_return'].ix[i - 2]
        dax_log_return_3 = log_return_data['DAX_log_return'].ix[i - 3]
        stoxx_log_return_1 = log_return_data['STOXX_log_return'].ix[i - 1]
        stoxx_log_return_2 = log_return_data['STOXX_log_return'].ix[i - 2]
        stoxx_log_return_3 = log_return_data['STOXX_log_return'].ix[i - 3]
        hkse_log_return_0 = log_return_data['HKSE_log_return'].ix[i]
        hkse_log_return_1 = log_return_data['HKSE_log_return'].ix[i - 1]
        hkse_log_return_2 = log_return_data['HKSE_log_return'].ix[i - 2]
        nikkei_log_return_0 = log_return_data['NIKKEI_log_return'].ix[i]
        nikkei_log_return_1 = log_return_data['NIKKEI_log_return'].ix[i - 1]
        nikkei_log_return_2 = log_return_data['NIKKEI_log_return'].ix[i - 2]
        sp500_log_return_0 = log_return_data['S&P500_log_return'].ix[i]
        sp500_log_return_1 = log_return_data['S&P500_log_return'].ix[i - 1]
        sp500_log_return_2 = log_return_data['S&P500_log_return'].ix[i - 2]

        training_test_data = training_test_data.append(
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
    predictors_tf = training_test_data[training_test_data.columns[2:]]
    classes_tf = training_test_data[training_test_data.columns[:2]]
    training_set_size = int(len(training_test_data) * 0.8)  # 80/20 sep of training/testing
    test_set_size = len(training_test_data) - training_set_size
    training_predictors_tf = predictors_tf[:training_set_size]
    training_classes_tf = classes_tf[:training_set_size]
    test_predictors_tf = predictors_tf[training_set_size:]
    test_classes_tf = classes_tf[training_set_size:]

    # Another model with two hidden layers
    sess = tf.Session()
    num_predictors = len(training_predictors_tf.columns)
    num_classes = len(training_classes_tf.columns)
    feature_data = tf.placeholder("float", [None, num_predictors])
    actual_classes = tf.placeholder("float", [None, num_classes])

    weights1 = tf.Variable(tf.truncated_normal([num_predictors, n_hidden_nodes], stddev=0.0001))
    biases1 = tf.Variable(tf.ones([n_hidden_nodes]))
    weights2 = tf.Variable(tf.truncated_normal([n_hidden_nodes, num_classes], stddev=0.0001))
    biases2 = tf.Variable(tf.ones([2]))

    hidden_layer = tf.nn.relu(tf.matmul(feature_data, weights1) + biases1)
    model = tf.nn.softmax(tf.matmul(hidden_layer, weights2) + biases2)
    cost = -tf.reduce_sum(actual_classes * tf.log(model))
    train_op1 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    init = tf.global_variables_initializer()
    sess.run(init)
    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(actual_classes, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    for i in range(1, 30001):
        sess.run(
            train_op1,
            feed_dict={
                feature_data: training_predictors_tf.values,
                actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)
            }
        )
        if i % 5000 == 0:
            print("node", n_hidden_nodes, "iter", i, sess.run(
                accuracy,
                feed_dict={
                    feature_data: training_predictors_tf.values,
                    actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)
                }
            ))
    feed_dict = {
        feature_data: test_predictors_tf.values,
        actual_classes: test_classes_tf.values.reshape(len(test_classes_tf.values), 2)
    }
    return tf_confusion_metrics(model, actual_classes, sess, feed_dict)


# node_indices = []
# accuracies = []
# f1scores = []
#
# for n_nodes in range(1, 100):
#     if n_nodes % 5 == 0:
#         f1_score, accuracy = run_model(n_nodes)
#         node_indices.append(n_nodes)
#         accuracies.append(accuracy)
#         f1scores.append(f1_score)
#
# np.savetxt("result_5-100.csv", np.array([node_indices, accuracies, f1scores]), delimiter=",")
