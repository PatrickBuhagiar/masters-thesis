import pandas as pd
import tensorflow as tf
from pymongo import MongoClient

from best_architecture import tf_confusion_metrics
from stock_model import load_data, divide_into_training_testing, organise_data, extract_market_directions, log_diff, \
    get_model_predictions
from toolbox import load_macroeconomic_data, get_lagged_macroeconomic_data, extract_macroeconomic_data, \
    prepare_macroeconomic_data

# MongoDB initialisation
client = MongoClient('localhost', 27017)
db = client['thesis']

# Load the stock data that will be used for training and testing the meta model. 
start = pd.datetime(2014, 0o1, 0o1)
end = pd.datetime(2018, 0o1, 0o1)
df = load_data(start, end)
log_return_data = log_diff(df)
extract_market_directions(log_return_data)
training_test_data = organise_data(log_return_data)

# Split data into training and testing
inputs = training_test_data[training_test_data.columns[2:]]
outputs = training_test_data[training_test_data.columns[:2]]

# get all stored models and get predictions
cursor = db.posts.find({})  # get all items
meta_inputs = pd.DataFrame()
_range = pd.date_range(start, end)
meta_inputs['Date'] = _range[:inputs.index.size]
meta_inputs.set_index('Date')

for item in cursor:
    date = item.get("_id")
    print(date)
    model_predictions = get_model_predictions(date, inputs)
    meta_inputs[date + "_predictions_0"] = model_predictions[:, 0]
    meta_inputs[date + "_predictions_1"] = model_predictions[:, 1]
    # meta_inputs.ix[model_predictions[:,0] >= 0.5, date + "_prediction"] = 1


# Load all macroeconomic data
prepare_macroeconomic_data(start, end, meta_inputs)

meta_inputs = meta_inputs.set_index('Date')
# concatenate all model outputs and macro data into one matrix + expected output


# divide into training and testing
test_outputs, test_inputs, training_outputs, training_inputs = divide_into_training_testing(meta_inputs, outputs,
                                                                                            len(meta_inputs))

# build meta model
sess = tf.Session()
n_hidden_nodes = 45
num_predictors = len(training_inputs.columns)
num_classes = len(training_outputs.columns)
feature_data = tf.placeholder("float", [None, num_predictors], name="feature_data")
actual_classes = tf.placeholder("float", [None, num_classes], name="actual_classes")

weights1 = tf.Variable(tf.truncated_normal([num_predictors, n_hidden_nodes], stddev=0.0001))
biases1 = tf.Variable(tf.ones([n_hidden_nodes]))
weights2 = tf.Variable(tf.truncated_normal([n_hidden_nodes, num_classes], stddev=0.0001))
biases2 = tf.Variable(tf.ones([2]))

hidden_layer = tf.nn.relu(tf.matmul(feature_data, weights1) + biases1)
model = tf.nn.softmax(tf.matmul(hidden_layer, weights2) + biases2)
cost = -tf.reduce_sum(actual_classes * tf.log(model))
train_op1 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
init = tf.global_variables_initializer()

# Run Model
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
    # Every 5000, we are going to print the current training accuracy of the model
    if i % 5000 == 0:
        print("start date", start, "iteration", i, sess.run(
            accuracy,
            feed_dict={
                feature_data: training_inputs.values,
                actual_classes: training_outputs.values.reshape(len(training_outputs.values), 2)
            }
        ))

feed_dict = {
    feature_data: test_inputs.values,
    actual_classes: test_outputs.values.reshape(len(test_outputs.values), 2)
}

# Calculate the F1 Score and Accuracy with the training set
f1_score, accuracy = tf_confusion_metrics(model, actual_classes, sess, feed_dict)
print(start, f1_score, accuracy)

# execute model
