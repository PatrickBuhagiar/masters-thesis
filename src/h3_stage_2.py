from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import pandas as pd
import tensorflow as tf

# Concurrency stuff
pool = ThreadPoolExecutor(20)
futures = []


def stringify(data: []):
    """
    convert data to string
    :param data: data
    :return:
    """
    return data.__str__()


def extract_index(filename, start_date, end_date, date_parse, dropna=True):
    """
    extract the index
    :param filename:
    :param start_date:
    :param end_date:
    :param date_parse:
    :param dropna:
    :return:
    """
    data = pd.read_csv(filename, parse_dates=['Date'], index_col='Date', date_parser=date_parse)
    # Fill missing dates and values
    all_days = pd.date_range(start_date, end_date, freq='D')
    data = data.reindex(all_days)
    ts = data['Close']
    if dropna:
        ts = ts.dropna()
    return ts


def prepare_data(start_date, end_date):
    """
    prepare the data for feed forward network
    :param start_date:
    :param end_date:
    :return:
    """
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    dateparse2 = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y')
    dateparse3 = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')

    # Prepare FTSE
    ftse_data = extract_index('../data/indices/^FTSE.csv', start_date, end_date, dateparse3)
    ftse_normalised = ftse_data / max(ftse_data)
    ftse_log = np.log(ftse_normalised / ftse_normalised.shift())
    ftse_log = ftse_log.dropna()

    # Prepare other stocks
    cac_data = extract_index('../data/indices/^FCHI.csv', start_date, end_date, dateparse)
    dax_data = extract_index('../data/indices/^GDAXI.csv', start_date, end_date, dateparse)
    sp500_data = extract_index('../data/indices/^GSPC.csv', start_date, end_date, dateparse2)
    n225_data = extract_index('../data/indices/^N225.csv', start_date, end_date, dateparse)
    stoxx_data = extract_index('../data/indices/^STOXX50E.csv', start_date, end_date, dateparse2)
    hkse_data = extract_index('../data/indices/^HSI.csv', start_date, end_date, dateparse)

    cac_normalised = cac_data / max(cac_data)
    dax_normalised = dax_data / max(dax_data)
    sp500_normalised = sp500_data / max(sp500_data)
    n225_normalised = n225_data / max(n225_data)
    stoxx_normalised = stoxx_data / max(stoxx_data)
    hkse_normalised = hkse_data / max(hkse_data)

    cac_log = np.log(cac_normalised / cac_normalised.shift())
    dax_log = np.log(dax_normalised / dax_normalised.shift())
    sp500_log = np.log(sp500_normalised / sp500_normalised.shift())
    n225_log = np.log(n225_normalised / n225_normalised.shift())
    stoxx_log = np.log(stoxx_normalised / stoxx_normalised.shift())
    hkse_log = np.log(hkse_normalised / hkse_normalised.shift())

    cac_log = cac_log.dropna()
    dax_log = dax_log.dropna()
    sp500_log = sp500_log.dropna()
    stoxx_log = stoxx_log.dropna()
    hkse_log = hkse_log.dropna()
    n225_log = n225_log.dropna()

    # Prepare directions
    directions = pd.DataFrame()
    directions['FTSE'] = ftse_log
    directions['UP'] = 0
    directions.ix[directions['FTSE'] >= 0, 'UP'] = 1
    directions['DOWN'] = 0
    directions.ix[directions['FTSE'] < 0, 'DOWN'] = 1

    data = pd.DataFrame(
        columns=['up', 'ftse_1', 'ftse_2', 'ftse_3', 'cac_1', 'cac_2', 'cac_3', 'dax_1', 'dax_2', 'dax_3', 'stoxx_1',
                 'stoxx_2', 'stoxx_3', 'sp500_1', 'sp500_2', 'sp500_3', 'hkse_0', 'hkse_1', 'hkse_2', 'n225_0',
                 'n225_1', 'n225_2']
    )

    dates = []
    for i in range(7, len(ftse_log)):
        up = directions['UP'].ix[i]

        date_0 = ftse_log.keys()[i]
        date_1 = ftse_log.keys()[i - 1]
        date_2 = ftse_log.keys()[i - 2]
        date_3 = ftse_log.keys()[i - 3]
        date_4 = ftse_log.keys()[i - 4]
        date_5 = ftse_log.keys()[i - 5]
        date_6 = ftse_log.keys()[i - 6]

        ftse_1 = ftse_log.ix[i - 1]
        ftse_2 = ftse_log.ix[i - 2]
        ftse_3 = ftse_log.ix[i - 3]

        cac_1 = 0
        cac_2 = 0
        cac_3 = 0
        if cac_log.keys().contains(date_3):
            cac_3 = cac_log.ix[date_3]
        elif cac_log.keys().contains(date_4):
            cac_3 = cac_log.ix[date_4]
        elif cac_log.keys().contains(date_5):
            cac_3 = cac_log.ix[date_5]
        else:
            cac_3 = cac_log.ix[date_6]

        if cac_log.keys().contains(date_2):
            cac_2 = cac_log.ix[date_2]
        else:
            cac_2 = cac_3

        if cac_log.keys().contains(date_1):
            cac_1 = cac_log.ix[date_1]
        else:
            cac_1 = cac_2

        dax_1 = 0
        dax_2 = 0
        dax_3 = 0
        if dax_log.keys().contains(date_3):
            dax_3 = dax_log.ix[date_3]
        elif dax_log.keys().contains(date_4):
            dax_3 = dax_log.ix[date_4]
        elif dax_log.keys().contains(date_5):
            dax_3 = dax_log.ix[date_5]
        else:
            dax_3 = dax_log.ix[date_6]

        if dax_log.keys().contains(date_2):
            dax_2 = dax_log.ix[date_2]
        else:
            dax_2 = dax_3

        if dax_log.keys().contains(date_1):
            dax_1 = dax_log.ix[date_1]
        else:
            dax_1 = dax_2

        stoxx_1 = 0
        stoxx_2 = 0
        stoxx_3 = 0
        if stoxx_log.keys().contains(date_3):
            stoxx_3 = stoxx_log.ix[date_3]
        elif stoxx_log.keys().contains(date_4):
            stoxx_3 = stoxx_log.ix[date_4]
        elif stoxx_log.keys().contains(date_5):
            stoxx_3 = stoxx_log.ix[date_5]
        else:
            stoxx_3 = stoxx_log.ix[date_6]

        if stoxx_log.keys().contains(date_2):
            stoxx_2 = stoxx_log.ix[date_2]
        else:
            stoxx_2 = stoxx_3

        if stoxx_log.keys().contains(date_1):
            stoxx_1 = stoxx_log.ix[date_1]
        else:
            stoxx_1 = stoxx_2

        sp500_1 = 0
        sp500_2 = 0
        sp500_3 = 0
        if sp500_log.keys().contains(date_3):
            sp500_3 = sp500_log.ix[date_3]
        elif sp500_log.keys().contains(date_4):
            sp500_3 = sp500_log.ix[date_4]
        elif sp500_log.keys().contains(date_5):
            sp500_3 = sp500_log.ix[date_5]
        else:
            sp500_3 = sp500_log.ix[date_6]

        if sp500_log.keys().contains(date_2):
            sp500_2 = sp500_log.ix[date_2]
        else:
            sp500_2 = sp500_3

        if sp500_log.keys().contains(date_1):
            sp500_1 = sp500_log.ix[date_1]
        else:
            sp500_1 = sp500_2

        hkse_0 = 0
        hkse_1 = 0
        hkse_2 = 0
        if hkse_log.keys().contains(date_2):
            hkse_2 = hkse_log.ix[date_2]
        elif hkse_log.keys().contains(date_3):
            hkse_2 = hkse_log.ix[date_3]
        elif hkse_log.keys().contains(date_4):
            hkse_2 = hkse_log.ix[date_4]
        else:
            hkse_2 = hkse_log.ix[date_5]

        if hkse_log.keys().contains(date_1):
            hkse_1 = hkse_log.ix[date_1]
        else:
            hkse_1 = hkse_2

        if hkse_log.keys().contains(date_0):
            hkse_0 = hkse_log.ix[date_0]
        else:
            hkse_0 = hkse_1

        n225_0 = 0
        n225_1 = 0
        n225_2 = 0
        if n225_log.keys().contains(date_2):
            n225_2 = n225_log.ix[date_2]
        elif n225_log.keys().contains(date_3):
            n225_2 = n225_log.ix[date_3]
        elif n225_log.keys().contains(date_4):
            n225_2 = n225_log.ix[date_4]
        elif n225_log.keys().contains(date_5):
            n225_2 = n225_log.ix[date_5]
        else:
            n225_2 = n225_log.ix[date_6]

        if n225_log.keys().contains(date_1):
            n225_1 = n225_log.ix[date_1]
        else:
            n225_1 = n225_2

        if n225_log.keys().contains(date_0):
            n225_0 = n225_log.ix[date_0]
        else:
            n225_0 = n225_1
        data = data.append(
            {
                'up': up,
                'ftse_1': ftse_1,
                'ftse_2': ftse_2,
                'ftse_3': ftse_3,
                'cac_1': cac_1,
                'cac_2': cac_2,
                'cac_3': cac_3,
                'dax_1': dax_1,
                'dax_2': dax_2,
                'dax_3': dax_3,
                'stoxx_1': stoxx_1,
                'stoxx_2': stoxx_2,
                'stoxx_3': stoxx_3,
                'sp500_1': sp500_1,
                'sp500_2': sp500_2,
                'sp500_3': sp500_3,
                'hkse_0': hkse_0,
                'hkse_1': hkse_1,
                'hkse_2': hkse_2,
                'n225_0': n225_0,
                'n225_1': n225_1,
                'n225_2': n225_2
            }, ignore_index=True
        )
        dates.append(ftse_data.index[i])
    inputs = data[data.columns[1:]]
    outputs = data[data.columns[:1]]
    return inputs, outputs, dates


def get_model_names():
    """
    get the names of the models, i.e. the starting date of the period it covers
    :return:
    """
    start_years = np.arange(2000, 2008, 1)
    start_dates = []
    file_names = []
    for year in start_years:
        start_dates.append(pd.datetime(year, 1, 1))
        start_dates.append(pd.datetime(year, 7, 1))
    for date in start_dates:
        file_names.append(date.date().__str__())
    return file_names


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


def get_model_predictions(filename, input):
    """
    The the predictions from the individual trained models
    :param filename:
    :param input:
    :return:
    """
    tf.reset_default_graph()
    # Load model and variables
    with tf.Session() as sess:
        name = filename[0] + filename[1] + filename[2] + filename[3] + filename[6]
        saver = tf.train.import_meta_graph("h3_models/" + filename + "/" + filename + ".meta")
        saver.restore(sess, tf.train.latest_checkpoint("h3_models/" + filename + "/"))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X_" + name + ":0")
        keep_prob = graph.get_tensor_by_name("keep_prob_" + name + ":0")
        feed_dict = {
            X: input.values,
            keep_prob: 1
        }

        predicted = graph.get_tensor_by_name("predicted_" + name + ":0")
        predictions = sess.run(tf.cast(tf.round(predicted), tf.int32), feed_dict)
        sess.close()
        return predictions


def convert_to_date(date):
    """
    Covert a given date String into a pandas datetime

    :param date: A date as a string. this can be in the following formats:
            - 2018Q1
            - 2018 JAN

    :return: a datetime
    """
    if date.__contains__("Q"):
        year, quarter = date.replace(" ", "").split("Q")
        if quarter == "1":
            return pd.datetime(int(year), 1, 1)
        elif quarter == "2":
            return pd.datetime(int(year), 4, 1)
        elif quarter == "3":
            return pd.datetime(int(year), 7, 1)
        else:
            return pd.datetime(int(year), 10, 1)
    else:
        year, month = date.split(" ")
        int_month = 0
        if month == "JAN":
            int_month = 1
        elif month == "FEB":
            int_month = 2
        elif month == "MAR":
            int_month = 3
        elif month == "APR":
            int_month = 4
        elif month == "MAY":
            int_month = 5
        elif month == "JUN":
            int_month = 6
        elif month == "JUL":
            int_month = 7
        elif month == "AUG":
            int_month = 8
        elif month == "SEP":
            int_month = 9
        elif month == "OCT":
            int_month = 10
        elif month == "NOV":
            int_month = 11
        elif month == "DEC":
            int_month = 12
        return pd.datetime(int(year), int_month, 1)


def extract_month(date):
    year, month = date.split(" ")
    int_month = 0
    if month == "JAN":
        int_month = 1
    elif month == "FEB":
        int_month = 2
    elif month == "MAR":
        int_month = 3
    elif month == "APR":
        int_month = 4
    elif month == "MAY":
        int_month = 5
    elif month == "JUN":
        int_month = 6
    elif month == "JUL":
        int_month = 7
    elif month == "AUG":
        int_month = 8
    elif month == "SEP":
        int_month = 9
    elif month == "OCT":
        int_month = 10
    elif month == "NOV":
        int_month = 11
    elif month == "DEC":
        int_month = 12
    return [int(year), int_month]


def extract_quarterly(date):
    if "Q" in date:
        year, quarter = date.replace(" ", "").split("Q")
        if quarter == "1":
            return [int(year), 1]
        elif quarter == "2":
            return [int(year), 2]
        elif quarter == "3":
            return [int(year), 3]
        else:
            return [int(year), 4]


def load_macroeconomic_data(filename, start_index, start, end, type='Q'):
    data = pd.read_csv(filename, index_col='Date')[start_index:]
    if type == 'D':
        all_days = pd.date_range(data.index[0], data.index[-1])
        data = data.reindex(all_days, fill_value=0)
        data.fillna(method='ffill')
    d = {}

    for date, row in data.iterrows():
        if type == 'Q' and extract_quarterly(date) is not None:
            if extract_quarterly(date)[0] >= (start.year - 2):
                d[(stringify(extract_quarterly(date)))] = float(row['Value'])
        elif type == 'M':
            d[stringify(extract_month(date))] = float(row['Value'])
        elif type == 'D':
            parsed_date = pd.to_datetime(date)
            d[parsed_date.__str__()] = float(row['Value'])
    return d


def extract_macroeconomic_data(filename, start_index, start, end, type='Q'):
    data = pd.read_csv(filename, index_col='Date')[start_index:]
    d = {'Date': [], 'Value': []}

    for index, row in data.iterrows():
        if type == 'Q':
            d['Date'].append(convert_to_date(index))
        elif type == 'D':
            d['Date'].append(pd.to_datetime(index))
        d['Value'].append(float(row['Value']))

    ts = pd.DataFrame(d)
    ts = ts.set_index('Date')
    all_days = pd.date_range(start - pd.DateOffset(months=1), end, freq='D')
    ts = ts.reindex(all_days)
    ts = ts.fillna(method='ffill')
    filtered_day_range = pd.date_range(start, end, freq='D')
    ts = ts.reindex(filtered_day_range)
    data.fillna(method='ffill')
    ts = ts.dropna()
    return ts


def get_lagged_macroeconomic_data(data, date: pd.datetime, type='Q'):
    if type == 'Q':
        # convert current date to Q1 and year, and return that quarter and the 3 previous one
        if 0 < date.month < 4:
            t_0 = data[stringify([date.year, 1])]
            t_1 = data[stringify([date.year - 1, 4])]
            t_2 = data[stringify([date.year - 1, 3])]
            t_3 = data[stringify([date.year - 1, 2])]
        elif 4 <= date.month < 7:
            t_0 = data[stringify([date.year, 2])]
            t_1 = data[stringify([date.year, 1])]
            t_2 = data[stringify([date.year - 1, 4])]
            t_3 = data[stringify([date.year - 1, 3])]
        elif 7 <= date.month < 10:
            t_0 = data[stringify([date.year, 3])]
            t_1 = data[stringify([date.year, 2])]
            t_2 = data[stringify([date.year, 1])]
            t_3 = data[stringify([date.year - 1, 4])]
        else:
            t_0 = data[stringify([date.year, 4])]
            t_1 = data[stringify([date.year, 3])]
            t_2 = data[stringify([date.year, 2])]
            t_3 = data[stringify([date.year, 1])]
        return [t_0, t_1, t_2, t_3]
    elif type == 'M':
        t_0 = data[stringify([date.year, date.month])]
        t_1_date = date - pd.DateOffset(months=3)
        t_1 = data[stringify([t_1_date.year, t_1_date.month])]
        t_2_date = date - pd.DateOffset(months=6)
        t_2 = data[stringify([t_2_date.year, t_2_date.month])]
        t_3_date = date - pd.DateOffset(months=12)
        t_3 = data[stringify([t_3_date.year, t_3_date.month])]
        return [t_0, t_1, t_2, t_3]  # convert that date to month, and return the current month, 3, 6, and 12 months ago
    elif type == 'D':
        t_0 = data.T[date].Value
        t_1 = data.T[(date - pd.DateOffset(months=3)).__str__()].Value
        t_2 = data.T[(date - pd.DateOffset(months=6)).__str__()].Value
        t_3 = data.T[(date - pd.DateOffset(months=12)).__str__()].Value
        return [t_0, t_1, t_2, t_3]


def prepare_macroeconomic_data(start, end, meta_inputs, dates):
    trade_balance_data = load_macroeconomic_data("../data/macroeconomics/GB/GB_BALANCE_OF_TRADE.csv", 90, start, end)
    gdp_data = load_macroeconomic_data("../data/macroeconomics/GB/GB_GDP.csv", 90, start, end)
    inflation_data = load_macroeconomic_data("../data/macroeconomics/GB/GB_INFLATION.csv", 22, start, end)
    interest_data = extract_macroeconomic_data("../data/macroeconomics/GB/GB_INTRST.csv", 1,
                                               start - pd.DateOffset(years=1),
                                               end, 'D')
    unemployment_data = load_macroeconomic_data("../data/macroeconomics/GB/GB_UNEMPLOYMENT.csv", 60, start, end)
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
    for date in dates:
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

        int_ = get_lagged_macroeconomic_data(interest_data, date, type='D')
        int_0.append(int_[0])
        int_1.append(int_[1])
        int_2.append(int_[2])
        int_3.append(int_[3])

        uem = get_lagged_macroeconomic_data(unemployment_data, date)
        uem_0.append(uem[0])
        uem_1.append(uem[1])
        uem_2.append(uem[2])
        uem_3.append(uem[3])
    meta_inputs['trade_balance_data_0'] = [
        (x - min(trade_balance_data.values())) / (max(trade_balance_data.values()) - min(trade_balance_data.values()))
        for x in tbd_0]
    meta_inputs['trade_balance_data_1'] = [
        (x - min(trade_balance_data.values())) / (max(trade_balance_data.values()) - min(trade_balance_data.values()))
        for x in tbd_1]
    meta_inputs['trade_balance_data_2'] = [
        (x - min(trade_balance_data.values())) / (max(trade_balance_data.values()) - min(trade_balance_data.values()))
        for x in tbd_2]
    # meta_inputs['trade_balance_data_3'] = [
    #     (x - min(trade_balance_data.values())) / (max(trade_balance_data.values()) - min(trade_balance_data.values()))
    #     for x in tbd_3]
    meta_inputs['gdp_data_0'] = [x / max(gdp_data.values()) for x in gdp_0]
    meta_inputs['gdp_data_1'] = [x / max(gdp_data.values()) for x in gdp_1]
    meta_inputs['gdp_data_2'] = [x / max(gdp_data.values()) for x in gdp_2]
    # meta_inputs['gdp_data_3'] = [x / max(gdp_data.values()) for x in gdp_3]
    meta_inputs['inflation_data_0'] = [x / max(inflation_data.values()) for x in inf_0]
    meta_inputs['inflation_data_1'] = [x / max(inflation_data.values()) for x in inf_1]
    meta_inputs['inflation_data_2'] = [x / max(inflation_data.values()) for x in inf_2]
    # meta_inputs['inflation_data_3'] = [x / max(inflation_data.values()) for x in inf_3]
    meta_inputs['interest_data_0'] = [x / max(interest_data.values) for x in int_0]
    meta_inputs['interest_data_1'] = [x / max(interest_data.values) for x in int_1]
    meta_inputs['interest_data_2'] = [x / max(interest_data.values) for x in int_2]
    # meta_inputs['interest_data_3'] = [x / max(interest_data.values) for x in int_3]
    meta_inputs['unemployment_data_0'] = [x / max(unemployment_data.values()) for x in uem_0]
    meta_inputs['unemployment_data_1'] = [x / max(unemployment_data.values()) for x in uem_1]
    meta_inputs['unemployment_data_2'] = [x / max(unemployment_data.values()) for x in uem_2]
    # meta_inputs['unemployment_data_3'] = [x / max(unemployment_data.values()) for x in uem_3]


def run(learn_rate, n_nodes, trn_inputs, trn_outputs, tst_inputs, tst_outputs, get_predictions=False):
    """
    Run with given parameters
    :param learn_rate:
    :param n_nodes:
    :param trn_inputs:
    :param trn_outputs:
    :param tst_inputs:
    :param tst_outputs:
    :param get_predictions:
    :return:
    """
    tf.reset_default_graph()
    feature_count = trn_inputs.shape[1]
    label_count = trn_outputs.shape[1]
    training_epochs = 3000

    cost_history = np.empty(shape=[1], dtype=float)
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
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    TP = tf.count_nonzero(tf.round(predicted) * Y)
    TN = tf.count_nonzero((tf.round(predicted) - 1) * (Y - 1))
    FP = tf.count_nonzero(tf.round(predicted) * (Y - 1))
    FN = tf.count_nonzero((tf.round(predicted) - 1) * Y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(training_epochs + 1):
            sess.run(optimizer, feed_dict={X: trn_inputs, Y: trn_outputs, keep_prob: 0.8})
            loss, _, acc = sess.run([cost, optimizer, accuracy],
                                    feed_dict={X: trn_inputs, Y: trn_outputs, keep_prob: 0.8})
            cost_history = np.append(cost_history, acc)

        if get_predictions:
            return sess.run([correct_pred, TP, TN, FP, FN], feed_dict={X: tst_inputs, Y: tst_outputs, keep_prob: 1})
        else:
            return sess.run([accuracy, TP, TN, FP, FN], feed_dict={X: tst_inputs, Y: tst_outputs, keep_prob: 1})


def process(j, nodes, lr_rates, accuracy_matrix, F1_score_matrix, trn_inputs, trn_outputs, tst_inputs, tst_outputs):
    """

    :param j:
    :param nodes: range of hidden nodes
    :param lr_rates: range of learning rates
    :param accuracy_matrix: accuracy results
    :param F1_score_matrix: f1 results
    :param trn_inputs: training inputs
    :param trn_outputs: training outputs
    :param tst_inputs: test inputs
    :param tst_outputs: test outputs
    :return:
    """
    n_nodes = nodes[j]
    for i in range(0, len(lr_rates)):
        learning_rate = lr_rates[i]
        acc = 0.0
        f1 = 0.0
        for k in range(0, 20):
            accuracy, TP, TN, FP, FN = run(learning_rate, n_nodes, trn_inputs, trn_outputs, tst_inputs,
                                           tst_outputs)
            acc += (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 += (2 * precision * recall) / (precision + recall)
            print("learning rate", "%.7f" % learning_rate, "n_nodes", n_nodes, "iter", k, "f1",
                  (2 * precision * recall) / (precision + recall), "accuracy", (TP + TN) / (TP + TN + FP + FN), TP, TN,
                  FP, FN)
        acc = acc / 20.0
        f1 = f1 / 20.0
        print("learning rate", "%.7f" % learning_rate, "n_nodes", n_nodes, "TOTAL", "f1",
              f1, "accuracy", acc)

        accuracy_matrix[j][i] = acc
        F1_score_matrix[j][i] = f1


def get_meta_inputs(inputs, dates, start, end):
    meta_inputs = pd.DataFrame()
    # load all stored models
    model_dates = get_model_names()
    for date in model_dates:
        model_predictions = get_model_predictions(date, inputs)
        meta_inputs[date + "_predictions"] = model_predictions[:, 0]
        print("processing model predictions for period", date)
    meta_inputs['ftse_1'] = inputs['ftse_1']
    meta_inputs['ftse_2'] = inputs['ftse_2']
    meta_inputs['ftse_3'] = inputs['ftse_3']
    # Load all macroeconomic data
    prepare_macroeconomic_data(start, end, meta_inputs, dates)
    return meta_inputs


if __name__ == '__main__':
    # Load the stock data that will be used for training and testing the meta model
    start = pd.datetime(2013, 1, 1)
    end = pd.datetime(2018, 1, 1)
    inputs, outputs, dates = prepare_data(start, end)

    meta_inputs = get_meta_inputs(inputs, dates, start, end)

    # split into training and testing
    test_outputs, test_inputs, training_outputs, training_inputs = divide_into_training_testing(meta_inputs, outputs,
                                                                                                len(meta_inputs))

    X = np.arange(40, 61, 1)  # number of nodes
    Y = np.arange(0.000001, 0.00001, 0.000001)  # learning rates
    accuracies = np.ones([len(X), len(Y)])
    f1s = np.ones([len(X), len(Y)])
    for j in range(0, len(X)):
        futures.append(
            pool.submit(process, j, X, Y, accuracies, f1s, training_inputs, training_outputs,
                        test_inputs,
                        test_outputs))

    wait(futures)
    np.savetxt("h3_accuracies_40-61_000005-00005.csv", accuracies, delimiter=",")
    np.savetxt("h3_f1s_40-61_000005-00005.csv", f1s, delimiter=",")
