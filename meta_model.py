import pandas as pd

from stock_model import load_data, divide_into_training_testing, organise_data, extract_market_directions, log_diff

start = pd.datetime(2014, 0o1, 0o1)
end = pd.datetime(2018, 0o1, 0o1)

# Load data
df = load_data(start, end)
log_return_data = log_diff(df)
extract_market_directions(log_return_data)
training_test_data = organise_data(log_return_data)

# Split data into training and testing
inputs = training_test_data[training_test_data.columns[2:]]
outputs = training_test_data[training_test_data.columns[:2]]
test_outputs, test_inputs, training_outputs, training_inputs = divide_into_training_testing(inputs, outputs,
                                                                                            len(training_test_data))

#get all stored dates in mongodb

# input data into models

# prepare macroeconomic data

# concatenate all model outputs and macro data into one matrix + expected output

#build meta model

#execute model

