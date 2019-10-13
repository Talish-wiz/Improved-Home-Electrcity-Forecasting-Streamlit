from numpy import split
from numpy import array
from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
from datetime import date
import calendar
import datetime

# split a univariate dataset into train/test sets
def split_dataset(data):
    # split into standard weeks
    train, test = data[1:-328], data[-328:-6]
    # restructure into windows of weekly data
    train = array(split(train, len(train)/7))
    test = array(split(test, len(test)/7))
    return train, test


# evaluate a single model
def evaluate_model(model_func, train, test):
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = model_func(history)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    predictions = array(predictions)
    return predictions

# convert windows of weekly multivariate data into a series of total power
def to_series(data):
    # extract just the total power from each week
    series = [week[:, 0] for week in data]
    # flatten into a single series
    series = array(series).flatten()
    return series

# arima forecast
def arima_forecast(history):
    # convert history into a univariate series
    series = to_series(history)
    # define the model
    model = ARIMA(series, order=(7,0,0))
    # fit the model
    model_fit = model.fit(disp=False)
    # make forecast
    yhat = model_fit.predict(len(series), len(series)+6)
    return yhat

# load the new file
def loadfile():
    dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
    # split into train and test
    train, test = split_dataset(dataset.values)
    # define the names and functions for the models we wish to evaluate
    models = dict()
    models['arima'] = arima_forecast
    for name, func in models.items():
        predict=evaluate_model(func, train, test)
    return predict

def working(l_date,predict):
    days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    f_date = date(2010, 1, 2)
    delta = l_date - f_date
    week=delta.days//7
    k=dict()
    for i in range(0,7):
        k[days[i]]=predict[week][i]
    return k
