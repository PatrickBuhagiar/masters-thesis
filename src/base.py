import numpy as np
import pandas as pd
import tensorflow as tf

# global variables
from best_architecture import tf_confusion_metrics
from stock_model import divide_into_training_testing
from toolbox import extract_index

start = pd.datetime(2008, 1, 1)
end = pd.datetime(2018, 1, 1)

# load FTSE and prepare data
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
            # 'down': down,
            'ftse_1': ftse_1,
            'ftse_2': ftse_2,
            'ftse_3': ftse_3,
            'ftse_4': ftse_4,
            'ftse_5': ftse_5
        }, ignore_index=True
    )

inputs = data[data.columns[1:]]
outputs = data[data.columns[:1]]

test_outputs, test_inputs, training_outputs, training_inputs = divide_into_training_testing(inputs, outputs, len(data))

feature_count = training_inputs.shape[1]
label_count = training_outputs.shape[1]
print(feature_count, label_count)

training_epochs = 3000
learning_rate = 0.005
n_hidden_nodes = 5
cost_history = np.empty(shape=[1], dtype=float)

X = tf.placeholder(tf.float32, [None, feature_count])
Y = tf.placeholder(tf.float32, [None, label_count])
is_training = tf.Variable(True, dtype=tf.bool)

initializer = tf.contrib.layers.xavier_initializer()
h0 = tf.layers.dense(X, n_hidden_nodes, activation=tf.nn.relu, kernel_initializer=initializer)
h0 = tf.nn.dropout(h0, 0.80)
h1 = tf.layers.dense(h0, label_count, activation=None)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=h1)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

predicted = tf.nn.sigmoid(h1)
correct_pred = tf.equal(tf.round(predicted), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(training_epochs + 1):
        sess.run(optimizer, feed_dict={X: training_inputs, Y: training_outputs})
        loss, _, acc = sess.run([cost, optimizer, accuracy], feed_dict={X: training_inputs, Y: training_outputs})
        cost_history = np.append(cost_history, acc)

        if step % 500 == 0:
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

    print('Test Accuracy:', sess.run([accuracy], feed_dict={X: test_inputs, Y: test_outputs}))
    test_predict_result = sess.run(tf.cast(tf.round(predicted), tf.int32), feed_dict={X: test_inputs})

    print("lol")
