import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas.plotting import autocorrelation_plot, scatter_matrix

from toolbox import load_indices

# Load Data
start = pd.datetime(2012, 2, 2)
end = pd.datetime(2017, 2, 2)
# GB_Balance_of_Trade = extract_macroeconomic_data("data/macroeconomics/GB/GB_BALANCE_OF_TRADE.csv", 222, start, end)
# GB_Gdp = extract_macroeconomic_data("data/macroeconomics/GB/GB_GDP.csv", 222, start, end)
# GB_Inflation = extract_macroeconomic_data("data/macroeconomics/GB/GB_INFLATION.csv", 69, start, end)
# GB_Interest = extract_macroeconomic_data("data/macroeconomics/GB/GB_INTRST.csv", 1, start, end, 'D')
# GB_Unemployment = extract_macroeconomic_data("data/macroeconomics/GB/GB_UNEMPLOYMENT.csv", 350, start, end)
indices = load_indices(start, end)
df = pd.DataFrame(index=indices['FTSE'].index)
# df['GB_TRADE_BLNC'] = GB_Balance_of_Trade
# df['GB_GDP'] = GB_Gdp
# df['GB_INFL'] = GB_Inflation
# df['GB_INTR'] = GB_Interest
# df['GB_UNEMP'] = GB_Unemployment
df['CAC'] = indices['CAC']
df['DAX'] = indices['DAX']
df['HKSE'] = indices['HKSE']
df['NIKKEI'] = indices['NIKKEI']
df['S&P500'] = indices['S&P500']
df['STOXX'] = indices['STOXX']
df['FTSE'] = indices['FTSE']

out = open("log.txt", "a")
print(df.describe())
out.write(df.describe())

df.plot()
plt.show()

# df['GB_TRADE_BLNC_scaled'] = df['GB_TRADE_BLNC'] / max(df['GB_TRADE_BLNC'])
# df['GB_GDP_scaled'] = df['GB_GDP'] / max(df['GB_GDP'])
# df['GB_INFL_scaled'] = df['GB_INFL'] / max(df['GB_INFL'])
# df['GB_INTR_scaled'] = df['GB_INTR'] / max(df['GB_INTR'])
# df['GB_UNEMP_scaled'] = df['GB_UNEMP'] / max(df['GB_UNEMP'])
df['CAC_scaled'] = df['CAC'] / max(df['CAC'])
df['DAX_scaled'] = df['DAX'] / max(df['DAX'])
df['HKSE_scaled'] = df['HKSE'] / max(df['HKSE'])
df['NIKKEI_scaled'] = df['NIKKEI'] / max(df['NIKKEI'])
df['S&P500_scaled'] = df['S&P500'] / max(df['S&P500'])
df['STOXX_scaled'] = df['STOXX'] / max(df['STOXX'])
df['FTSE_scaled'] = df['FTSE'] / max(df['FTSE'])

_ = pd.concat([  # df['GB_TRADE_BLNC_scaled'],
    # df['GB_GDP_scaled'],
    # df['GB_INFL_scaled'],
    # df['GB_INTR_scaled'],
    # df['GB_UNEMP_scaled'],
    df['CAC_scaled'],
    df['DAX_scaled'],
    df['HKSE_scaled'],
    df['NIKKEI_scaled'],
    df['S&P500_scaled'],
    df['STOXX_scaled'],
    df['FTSE_scaled']], axis=1).plot()
plt.show()

fig = plt.figure()

# _ = autocorrelation_plot(df['GB_TRADE_BLNC_scaled'], label="Trade Balance")
# _ = autocorrelation_plot(df['GB_GDP_scaled'], label="GDP")
# _ = autocorrelation_plot(df['GB_INFL_scaled'], label="INFLATION")
# _ = autocorrelation_plot(df['GB_INTR_scaled'], label="INTEREST")
# _ = autocorrelation_plot(df['GB_UNEMP_scaled'], label="UNEMPLOYMENT")
_ = autocorrelation_plot(df['CAC_scaled'], label="CAC")
_ = autocorrelation_plot(df['DAX_scaled'], label="DAX")
_ = autocorrelation_plot(df['HKSE_scaled'], label="HKSE")
_ = autocorrelation_plot(df['NIKKEI_scaled'], label="NIKKEI")
_ = autocorrelation_plot(df['S&P500_scaled'], label="S&P500")
_ = autocorrelation_plot(df['STOXX_scaled'], label="STOXX")
_ = autocorrelation_plot(df['FTSE_scaled'], label="FTSE")
_ = plt.legend(loc='upper right')

plt.show()

_ = scatter_matrix(pd.concat([  # df['GB_TRADE_BLNC_scaled'],
    # df['GB_GDP_scaled'],
    # df['GB_INFL_scaled'],
    # df['GB_INTR_scaled'],
    # df['GB_UNEMP_scaled'],
    df['CAC_scaled'],
    df['DAX_scaled'],
    df['HKSE_scaled'],
    df['NIKKEI_scaled'],
    df['S&P500_scaled'],
    df['STOXX_scaled'],
    df['FTSE_scaled']], axis=1), diagonal='kde')

plt.show()

log_return_data = pd.DataFrame()

log_return_data['CAC_log_return'] = np.log(df['CAC'] / df['CAC'].shift())
log_return_data['DAX_log_return'] = np.log(df['DAX'] / df['DAX'].shift())
log_return_data['HKSE_log_return'] = np.log(df['HKSE'] / df['HKSE'].shift())
log_return_data['NIKKEI_log_return'] = np.log(df['NIKKEI'] / df['NIKKEI'].shift())
log_return_data['S&P500_log_return'] = np.log(df['S&P500'] / df['S&P500'].shift())
log_return_data['STOXX_log_return'] = np.log(df['STOXX'] / df['STOXX'].shift())
log_return_data['FTSE_log_return'] = np.log(df['FTSE'] / df['FTSE'].shift())

print(log_return_data.describe())

_ = pd.concat([log_return_data['CAC_log_return'],
               log_return_data['DAX_log_return'],
               log_return_data['HKSE_log_return'],
               log_return_data['NIKKEI_log_return'],
               log_return_data['S&P500_log_return'],
               log_return_data['STOXX_log_return'],
               log_return_data['FTSE_log_return']], axis=1).plot()
plt.show()

fig = plt.figure()
_ = autocorrelation_plot(log_return_data['CAC_log_return'], label="CAC")
_ = autocorrelation_plot(log_return_data['DAX_log_return'], label="DAX")
_ = autocorrelation_plot(log_return_data['HKSE_log_return'], label="HKSE")
_ = autocorrelation_plot(log_return_data['NIKKEI_log_return'], label="NIKKEI")
_ = autocorrelation_plot(log_return_data['S&P500_log_return'], label="S&P500")
_ = autocorrelation_plot(log_return_data['STOXX_log_return'], label="STOXX")
_ = autocorrelation_plot(log_return_data['FTSE_log_return'], label="FTSE")
_ = plt.legend(loc='upper right')

plt.show()

_ = scatter_matrix(log_return_data, diagonal='kde')

plt.show()

tmp = pd.DataFrame()
tmp['FTSE_0'] = log_return_data['FTSE_log_return']
tmp['CAC_1'] = log_return_data['CAC_log_return'].shift()
tmp['DAX_1'] = log_return_data['DAX_log_return'].shift()
tmp['STOXX_1'] = log_return_data['STOXX_log_return'].shift()
tmp['HKSE_0'] = log_return_data['HKSE_log_return']
tmp['NIKKEI_0'] = log_return_data['NIKKEI_log_return']
tmp['S&P500_0'] = log_return_data['S&P500_log_return']
print(tmp.corr()['FTSE_0'])

# we will predict whether the FTSE will go up or down

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

print(training_test_data.describe())

predictors_tf = training_test_data[training_test_data.columns[2:]]
classes_tf = training_test_data[training_test_data.columns[:2]]

training_set_size = int(len(training_test_data) * 0.8)  # 80/20 sep of training/testing
test_set_size = len(training_test_data) - training_set_size

training_predictors_tf = predictors_tf[:training_set_size]
training_classes_tf = classes_tf[:training_set_size]
test_predictors_tf = predictors_tf[training_set_size:]
test_classes_tf = classes_tf[training_set_size:]

print(training_predictors_tf.describe())
print(test_predictors_tf.describe())


def tf_confusion_metrics(model, actual_classes, session, feed_dict):
    predictions = tf.argmax(model, 1)
    actuals = tf.argmax(actual_classes, 1)

    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    tp_op = tf.reduce_sum(
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
    fpr = float(fp) / (float(tp) + float(fn))

    accuracy = (float(tp) + float(tn)) / (float(tp) + float(fn) + float(tn))

    recall = tpr
    precision = float(tp) / (float(tp) + float(fp))

    f1_score = (2 * (precision * recall)) / (precision + recall)

    print('Precision = ', precision)
    print('Recall = ', recall)
    print('F1_Score = ', f1_score)
    print('Accuracy = ', accuracy)


sess = tf.Session()

# Define variables for the number of predictors and number of classes to remove magic numbers from our code
num_predictors = len(training_predictors_tf.columns)
num_classes = len(training_classes_tf.columns)

# Define placeholders for the data we feed into the process - feature data and actual classes.
feature_data = tf.placeholder("float", [None, num_predictors])
actual_classes = tf.placeholder("float", [None, num_classes])

# Define a matrix of weights and initialize it with some small random values
weights = tf.Variable(tf.truncated_normal([num_predictors, num_classes], stddev=0.0001))
biases = tf.Variable(tf.ones([num_classes]))

# Define our model...
# Here we take a softmax regression of the product of our feature data and weights
model = tf.nn.softmax(tf.matmul(feature_data, weights) + biases)

# Define a cost function (cross entropy)
cost = -tf.reduce_sum(actual_classes * tf.log(model))

# Define a training
training_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

init = tf.global_variables_initializer()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(actual_classes, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

for i in range(1, 30001):
    sess.run(
        training_step,
        feed_dict={
            feature_data: training_predictors_tf.values,
            actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)
        }
    )
    if i % 5000 == 0:
        print(i, sess.run(
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

tf_confusion_metrics(model, actual_classes, sess, feed_dict)

# Another model with two hidden layers
sess1 = tf.Session()
num_predictors = len(training_predictors_tf.columns)
num_classes = len(training_classes_tf.columns)

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
sess1.run(init)

correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(actual_classes, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

for i in range(1, 30001):
    sess1.run(
        train_op1,
        feed_dict={
            feature_data: training_predictors_tf.values,
            actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)
        }
    )
    if i % 5000 == 0:
        print(i, sess1.run(
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

tf_confusion_metrics(model, actual_classes, sess1, feed_dict)
out.close()