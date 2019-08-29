#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 22:57:54 2019

@author: weichi
"""
import datetime as dt
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics, model_selection
from sklearn.model_selection import train_test_split

# Reconstruct time infomation by `month`, `day`, and `hour`

def get_time(x):
    time_str = '2019 %d %d %d' % (x[0], x[1], x[2])
    taipei_tz = pytz.timezone('Asia/Taipei')
    time = dt.datetime.strptime(time_str, '%Y %m %d %H').replace(tzinfo=taipei_tz)
    return time

# check the next row is the next hour or not. 
# If it is not, the `pm2.5_next_hour` column will be given NaN.

def check_next_hour(x):
    one_hour = dt.timedelta(hours=1)
    if x[2] - x[1] == one_hour:
        return x[0]
    return np.nan

#%% read data from saved file
df5 = pd.read_csv('complete_data_5.csv', index_col=0)

#%% conbine date information
df5['date'] = df5[['month', 'day', 'hour']].apply(get_time, axis=1)

#%% plot
# Add explicitly converter
pd.plotting.register_matplotlib_converters()
# Plt
plt.figure(figsize=(12, 7))
plt.scatter(df5['date'], df5['pm2.5'])

#%% shift and append previous 1~5 hours data as columns next to original dataframe, maybe 10 better
titles = ['pm2.5', 'temp', 'humidity', 'ws', 'wd', 'precp']
for i in range(1, 8):
    for item in titles:
        title = item + '_' + str(i)
        df5[title] = df5[item].shift(periods=i)

#df5['pm2.5_shift_1'] = df5['pm2.5'].shift(periods=-1)
#%% drop nan, date column and reset index
df5 = df5.dropna(axis=0)
df5 = df5.reset_index(drop=True)
date = df5['date']
df5 = df5.drop(['date', 'pm1.0', 'pm10.0', 'month', 'day', 'hour'], axis=1) 

#%% Normalization
std = df5.std()
mean = df5.mean()
df5 = (df5 - mean) / std
df5['date'] = date
#%% shift pm2.5 data to get the next x hour
shift_amount = 1
df5['pm2.5_shift_1'] = df5['pm2.5'].shift(-shift_amount)
df5['time_shift_1'] = df5['date'].shift(-shift_amount)
#%%
df5 = df5.dropna(axis=0)
#date = date.drop(date.index[date.index.size-shift_amount:date.index.size])
df5 = df5.reset_index(drop=True)
#%%
pm_shift_1 = df5['pm2.5_shift_1']
time_shift_1 = df5['time_shift_1']
date = df5['date']
df5 = df5.drop(['date', 'pm2.5_shift_1', 'time_shift_1'], axis=1) 
#%% split pm2.5 data as y and remain as X
y = pm_shift_1.copy()
original_pm25 = pm_shift_1.copy()
X = df5.copy()

#%%
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)
#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)

#%% Divide training set and test set
bound = int(len(df5)*0.85)

X_train = X[:bound]
X_test = X[bound:]
y_train = y[:bound]
y_test = y[bound:]
date = date[bound:]
time_shift_1 = time_shift_1[bound:]
original_pm25 = original_pm25[bound:]
#%% Fit the model
model = linear_model.LinearRegression(normalize=True)
model.fit(X_train, y_train)

#%% See the coefficients of our model
a = model.coef_
b = model.intercept_
print(a)
print(b)

for i in range(len(X_train.columns)):
    print('Coefficient for %10s:\t%s' % (X_train.columns[i], model.coef_[i]))

#%% Calculate mean squared error for training set & test set
predict_train_y = model.predict(X_train)
predict_y = model.predict(X_test)

train_mse = metrics.mean_squared_error(y_train, predict_train_y)
test_mse = metrics.mean_squared_error(y_test, predict_y)

print('Train MSE:\t %s' % train_mse)
print('Test MSE:\t %s' % test_mse)

#%% denormalization
predict_y_plot = predict_y * std['pm2.5'] + mean['pm2.5']
y_test_plot = y_test * std['pm2.5'] + mean['pm2.5']
original_pm25_plot = original_pm25 * std['pm2.5'] + mean['pm2.5']

#%%
# Add explicitly converter
pd.plotting.register_matplotlib_converters()
# Plt
plt.figure(figsize=(12, 7))
plt.plot(time_shift_1, original_pm25_plot, label='actual values')
plt.plot(time_shift_1, predict_y_plot, label='predict values')
plt.legend()
fig = plt.gcf()
plt.show()
fig.savefig('output.png')




