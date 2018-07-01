import pandas as pd
from pymongo import MongoClient

from stock_model import load_data, divide_into_training_testing, organise_data, extract_market_directions, log_diff, \
    get_model_predictions, load_model

# MongoDB
from toolbox import load_macroeconomic_data, get_lagged_macroeconomic_data, extract_macroeconomic_data

client = MongoClient('localhost', 27017)
db = client['thesis']

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

# get all stored models and get predictions
cursor = db.posts.find({})  # get all items
dates = []
meta_inputs = pd.DataFrame()
range = pd.date_range(start, end)
meta_inputs['Date'] = range[:training_inputs.index.size]
meta_inputs.set_index('Date')
model_predictions = {}

for item in cursor:
    date = item.get("_id")
    dates.append(date)
    print(date)
    model_predictions[date + '_predictions'] = get_model_predictions(date, training_inputs)
    meta_inputs[date + "_predictions_0"] = model_predictions[date + '_predictions'][:, 0]
    meta_inputs[date + "_predictions_1"] = model_predictions[date + '_predictions'][:, 1]


# Load all macroeconomic data
trade_balance_data = load_macroeconomic_data("data/macroeconomics/GB/GB_BALANCE_OF_TRADE.csv", 90)
gdp_data = load_macroeconomic_data("data/macroeconomics/GB/GB_GDP.csv", 90)
inflation_data = load_macroeconomic_data("data/macroeconomics/GB/GB_INFLATION.csv", 22)
interest_data = extract_macroeconomic_data("data/macroeconomics/GB/GB_INTRST.csv", 1, start - pd.DateOffset(years=2), end, 'D')
unemployment_data = load_macroeconomic_data("data/macroeconomics/GB/GB_UNEMPLOYMENT.csv", 60)

tbd_0 = []
tbd_1 = []
tbd_2 = []
tbd_3 = []

gdp_0 = []
gdp_1 = []
gdp_2 = []
gdp_3 = []

inf_0 = []
inf_1 = []
inf_2 = []
inf_3 = []

int_0 = []
int_1 = []
int_2 = []
int_3 = []

uem_0 = []
uem_1 = []
uem_2 = []
uem_3 = []

for row in meta_inputs.iterrows():
    date = row[1].Date
    tbd = get_lagged_macroeconomic_data(trade_balance_data, date)
    tbd_0.append(tbd[0])
    tbd_1.append(tbd[1])
    tbd_2.append(tbd[2])
    tbd_3.append(tbd[3])

    gdp = get_lagged_macroeconomic_data(gdp_data, date)
    gdp_0.append(gdp[0])
    gdp_1.append(gdp[1])
    gdp_2.append(gdp[2])
    gdp_3.append(gdp[3])

    inf = get_lagged_macroeconomic_data(inflation_data, date)
    inf_0.append(inf[0])
    inf_1.append(inf[1])
    inf_2.append(inf[2])
    inf_3.append(inf[3])

    int = get_lagged_macroeconomic_data(interest_data, date, type='D')
    int_0.append(int[0])
    int_1.append(int[1])
    int_2.append(int[2])
    int_3.append(int[3])

    uem = get_lagged_macroeconomic_data(unemployment_data, date)
    uem_0.append(uem[0])
    uem_1.append(uem[1])
    uem_2.append(uem[2])
    uem_3.append(uem[3])

meta_inputs['trade_balance_data_0'] = tbd_0
meta_inputs['trade_balance_data_1'] = tbd_1
meta_inputs['trade_balance_data_2'] = tbd_2
meta_inputs['trade_balance_data_3'] = tbd_3

meta_inputs['gdp_data_0'] = gdp_0
meta_inputs['gdp_data_1'] = gdp_1
meta_inputs['gdp_data_2'] = gdp_2
meta_inputs['gdp_data_3'] = gdp_3

meta_inputs['inflation_data_0'] = inf_0
meta_inputs['inflation_data_1'] = inf_1
meta_inputs['inflation_data_2'] = inf_2
meta_inputs['inflation_data_3'] = inf_3

meta_inputs['interest_data_0'] = int_0
meta_inputs['interest_data_1'] = int_1
meta_inputs['interest_data_2'] = int_2
meta_inputs['interest_data_3'] = int_3

meta_inputs['unemployment_data_0'] = uem_0
meta_inputs['unemployment_data_1'] = uem_1
meta_inputs['unemployment_data_2'] = uem_2
meta_inputs['unemployment_data_3'] = uem_3

meta_inputs = meta_inputs.set_index('Date')
# concatenate all model outputs and macro data into one matrix + expected output

# build meta model

# execute model
