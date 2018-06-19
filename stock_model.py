import numpy as np
import pandas as pd
import tensorflow as tf

from toolbox import load_indices

# Load Data
start = pd.datetime(2012, 2, 2)
end = pd.datetime(2017, 2, 2)


def normalise_data(df):
    df['CAC_scaled'] = df['CAC'] / max(df['CAC'])
    df['DAX_scaled'] = df['DAX'] / max(df['DAX'])
    df['HKSE_scaled'] = df['HKSE'] / max(df['HKSE'])
    df['NIKKEI_scaled'] = df['NIKKEI'] / max(df['NIKKEI'])
    df['S&P500_scaled'] = df['S&P500'] / max(df['S&P500'])
    df['STOXX_scaled'] = df['STOXX'] / max(df['STOXX'])
    df['FTSE_scaled'] = df['FTSE'] / max(df['FTSE'])


def load_data(end, start):
    indices = load_indices(start, end)
    df = pd.DataFrame(index=indices['FTSE'].index)
    df['CAC'] = indices['CAC']
    df['DAX'] = indices['DAX']
    df['HKSE'] = indices['HKSE']
    df['NIKKEI'] = indices['NIKKEI']
    df['S&P500'] = indices['S&P500']
    df['STOXX'] = indices['STOXX']
    df['FTSE'] = indices['FTSE']
    normalise_data(df)
    return df


def log_diff(df):
    log_return_data = pd.DataFrame()
    log_return_data['CAC_log_return'] = np.log(df['CAC'] / df['CAC'].shift())
    log_return_data['DAX_log_return'] = np.log(df['DAX'] / df['DAX'].shift())
    log_return_data['HKSE_log_return'] = np.log(df['HKSE'] / df['HKSE'].shift())
    log_return_data['NIKKEI_log_return'] = np.log(df['NIKKEI'] / df['NIKKEI'].shift())
    log_return_data['S&P500_log_return'] = np.log(df['S&P500'] / df['S&P500'].shift())
    log_return_data['STOXX_log_return'] = np.log(df['STOXX'] / df['STOXX'].shift())
    log_return_data['FTSE_log_return'] = np.log(df['FTSE'] / df['FTSE'].shift())
    return log_return_data


def extract_market_directions(log_return_data):
    log_return_data['ftse_log_return_positive'] = 0
    log_return_data.ix[log_return_data['FTSE_log_return'] >= 0, 'ftse_log_return_positive'] = 1
    log_return_data['ftse_log_return_negative'] = 0
    log_return_data.ix[log_return_data['FTSE_log_return'] < 0, 'ftse_log_return_negative'] = 1


def organise_data(log_return_data):
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


def divide_into_training_testing(inputs_tf, output_classes_tf, n):
    training_set_size = int(n * 0.8)  # 80/20 sep of training/testing
    training_predictors_tf = inputs_tf[:training_set_size]
    training_classes_tf = output_classes_tf[:training_set_size]
    test_predictors_tf = inputs_tf[training_set_size:]
    test_classes_tf = output_classes_tf[training_set_size:]
    return test_classes_tf, test_predictors_tf, training_classes_tf, training_predictors_tf


def get_model(start, end):
    df = load_data(end, start)
    log_return_data = log_diff(df)
    extract_market_directions(log_return_data)

    training_test_data = organise_data(log_return_data)

    inputs = training_test_data[training_test_data.columns[2:]]
    outputs = training_test_data[training_test_data.columns[:2]]

    test_outputs, test_inputs, training_outputs, training_inputs = divide_into_training_testing(
        inputs, outputs, len(training_test_data))

    sess = tf.Session()
    num_predictors = len(training_inputs.columns)
    num_classes = len(training_outputs.columns)

    feature_data = tf.placeholder("float", [None, num_predictors])
    actual_classes = tf.placeholder("float", [None, num_classes])

    weights1 = tf.Variable(tf.truncated_normal([num_predictors, 50], stddev=0.0001))
    biases1 = tf.Variable(tf.ones([50]))

    weights2 = tf.Variable(tf.truncated_normal([50, num_predictors + 1], stddev=0.0001))
    biases2 = tf.Variable(tf.ones([num_predictors + 1]))

    weights3 = tf.Variable(tf.truncated_normal([num_predictors + 1, num_classes], stddev=0.0001))
    biases3 = tf.Variable(tf.ones([2]))

    hidden_layer_1 = tf.nn.relu(tf.matmul(feature_data, weights1) + biases1)
    hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, weights2) + biases2)
    model = tf.nn.softmax(tf.matmul(hidden_layer_2, weights3) + biases3)

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
                feature_data: training_inputs.values,
                actual_classes: training_outputs.values.reshape(len(training_outputs.values), 2)
            }
        )
        if i % 5000 == 0:
            print(i, sess.run(
                accuracy,
                feed_dict={
                    feature_data: training_inputs.values,
                    actual_classes: training_outputs.values.reshape(len(training_outputs.values), 2)
                }
            ))

    test_dict = {
        feature_data: test_inputs.values,
        actual_classes: test_outputs.values.reshape(len(test_outputs.values), 2)
    }

    return model, actual_classes, sess, test_dict
