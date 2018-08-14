import numpy as np
import pandas as pd
import tensorflow as tf

# global variables
from best_architecture import tf_confusion_metrics
from stock_model import divide_into_training_testing
from toolbox import extract_index

start = pd.datetime(2013, 1, 1)
end = pd.datetime(2018, 1, 1)

n_hidden_nodes = 4

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
    columns=['up', 'down', 'ftse_1', 'ftse_2', 'ftse_3']
)
for i in range(7, len(ts_log)):
    up = directions['UP'].ix[i]
    down = directions['DOWN'].ix[i]
    ftse_1 = ts_log.ix[i - 1]
    ftse_2 = ts_log.ix[i - 2]
    ftse_3 = ts_log.ix[i - 3]
    data = data.append(
        {
            'up': up,
            'down': down,
            'ftse_1': ftse_1,
            'ftse_2': ftse_2,
            'ftse_3': ftse_3
        }, ignore_index=True
    )

inputs = data[data.columns[2:]]
outputs = data[data.columns[:2]]

test_outputs, test_inputs, training_outputs, training_inputs = divide_into_training_testing(inputs, outputs, len(data))

# Build model
sess = tf.Session()
num_predictors = len(training_inputs.columns)
num_classes = len(training_outputs.columns)
feature_data = tf.placeholder("float", [None, num_predictors], name="feature_data")
actual_classes = tf.placeholder("float", [None, num_classes], name="actual_classes")

weights1 = tf.Variable(tf.truncated_normal([num_predictors, n_hidden_nodes], stddev=0.0001), name="w1")
biases1 = tf.Variable(tf.ones([n_hidden_nodes]), name="b1")
weights2 = tf.Variable(tf.truncated_normal([n_hidden_nodes, num_classes], stddev=0.0001), name="w2")
biases2 = tf.Variable(tf.ones([2]), name="b2")

hidden_layer = tf.nn.relu(tf.matmul(feature_data, weights1) + biases1, name="h")
model = tf.nn.softmax(tf.matmul(hidden_layer, weights2) + biases2, name="model")
cost = -tf.reduce_sum(actual_classes * tf.log(model), name="cost")
train_op1 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
init = tf.global_variables_initializer()

sess.run(init)

binary_prediction = tf.greater_equal(model, 0.5)

correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(actual_classes, 1), name="prediction")
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")
for i in range(1, 30001):
    sess.run(
        train_op1,
        feed_dict={
            feature_data: training_inputs.values,
            actual_classes: training_outputs.values.reshape(len(training_outputs.values), 2)
        }
    )
    # Every 5000, we are going to print the current training accuracy of the model
    if i % 5000 == 0:
        print("start date", start, "iteration", i, sess.run(
            binary_prediction,
            feed_dict={
                feature_data: training_inputs.values,
                actual_classes: training_outputs.values.reshape(len(training_outputs.values), 2)
            }
        ))

feed_dict = {
    feature_data: test_inputs.values,
    actual_classes: test_outputs.values.reshape(len(test_outputs.values), 2)
}

predicted = tf.argmax(model, 1)
actual = tf.argmax(actual_classes, 1)

TP = tf.count_nonzero(tf.multiply(predicted, actual))
TN = tf.count_nonzero(tf.multiply(tf.subtract(predicted, 1), tf.subtract(actual, 1)))
FP = tf.count_nonzero(tf.multiply(predicted, tf.subtract(actual, 1)))
FN = tf.count_nonzero(tf.multiply(tf.subtract(predicted, 1), actual))

tp, tn, fp, fn = sess.run(
    [TP, TN, FP, FN],
    feed_dict
)

precision = tp/ (tp + fp)
recall = tp / (tp + fn)
f1 = 2.0 * precision * recall / (precision + recall)
accuracy = (tp + tn) / (tp + fn + tn + fp)


# Calculate the F1 Score and Accuracy with the training set
f1_score1, accuracy1 = tf_confusion_metrics(model, actual_classes, sess, feed_dict)
print(f1, accuracy)
