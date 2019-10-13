import streamlit as st
import numpy
import pandas
import time
import datetime
from matplotlib import pyplot
import project
import calendar
from pandas import read_csv

def jhaki():
    my_bar=st.progress(0)
    status_text=st.empty()
    for percent in range(101):
        if percent != 100:
            status_text.text(f'Loading.....: {percent}%')
            my_bar.progress(percent + 1)
        else:
            status_text.text(f'Completed !!: {percent}%')
        time.sleep(0.1)


@st.cache
def loadfile():
    dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
    # split into train and test
    train, test = project.split_dataset(dataset.values)
    # define the names and functions for the models we wish to evaluate
    models = dict()
    models['arima'] = project.arima_forecast
    for name, func in models.items():
        predict=project.evaluate_model(func, train, test)
    return predict

def display_graph(y):
    x = ['Sunday', 'Monday', 'Tuesday', 'Wednesday','Thursday', 'Friday', 'Saturday']
    pyplot.plot(x,y,marker='o',label='ARIMA')
    pyplot.legend()
    pyplot.title("Graph of full standard week having your date")
    pyplot.xlabel("(Weekdays)")
    pyplot.ylabel("(Units consumed)")
    st.pyplot()

def input_date(predict):
    f=datetime.date(2010, 1, 3)
    l=datetime.date(2010, 11, 20)
    d=datetime.date(2010, 1, 2)
    l_date=st.date_input('Select Date (Date must be between 03-01-2010 to 20-11-2010)',datetime.date(2010, 1, 2))
    if l_date==d:
        st.write("")
    elif(l_date>=f and l_date<=l):
        st.subheader('Result')
        k=project.working(l_date,predict)
        st.write(f"Predicted Household Consumption on {l_date}({calendar.day_name[l_date.weekday()]}): {k[calendar.day_name[l_date.weekday()]]}")
        y=[]
        for i in k:
            y.append(k[i])
        #st.text("Graph of given date's week:")
        display_graph(y)
    else:
        st.error("Please insert date between 03-01-2010 to 20-11-2010")


st.title('Improved Home Electricity Forecasting')
st.subheader('Project Objective:')
st.text('The aim of the project is to carry out a short term forecast on electricity consumption of\na single home using ARIMA model')
st.subheader('Dataset Information:')
st.text('Dataset is take from UCI Machine Learning repositiory.\nThis archive contains 2075259 measurements gathered in\na house located in Sceaux(7km of Paris, France) between\nDecember 2006 and November 2010 (47 months)\n')
if st.button('Load Dataset'):
    jhaki()
    dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
    st.write(dataset.head(n=20))
predict=loadfile()
input_date(predict)
