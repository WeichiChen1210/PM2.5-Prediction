#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 22:57:54 2019

@author: weichi
"""
import datetime as dt
import requests
import json
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics, model_selection
from scipy import stats

# Reconstruct time infomation by `month`, `day`, and `hour`

def get_time(x):
    time_str = '2019 %d %d %d' % (x[0], x[1], x[2])
    taipei_tz = pytz.timezone('Asia/Taipei')
    time = dt.datetime.strptime(time_str, '%Y %m %d %H').replace(tzinfo=taipei_tz)
    return time
#%% Find the lost days
#count = 1
#count_hour = 0
#lost_day = {'6': [], '7': [], '8': []}
#temp_list = []
#for index, rows in df5mean.iterrows():
#    month = rows['month']
#    day = rows['day']
#    hour = rows['hour']
#    if day != count:        
#        if day == (count + 1):
#            count = day
#        elif day < count:            
#            # print(day, count)
#            count = 1
#            while count != day:
#                temp_list.append(count)
#                count += 1
#            tmp = str(int(month))
#            lost_day[tmp] += temp_list
#            temp_list.clear()
#            count = day
#        else:
#            print(day, count)
#            count += 1
#            while count != day:
#                temp_list.append(count)
#                count += 1
#            tmp = str(int(month))
#            lost_day[tmp] += temp_list
#            temp_list.clear()
#            count = day     
#        
#    if hour != count_hour:
#        
#        #print(month, day, hour)
#        count_hour = hour + 1
#    else:
#        count_hour += 1
#    if count_hour > 23:
#        count_hour = 0
#%% read data from saved file
df5mean = pd.read_csv('complete_data_5.csv', index_col=0)

#%%
df5mean['time'] = df5mean[['month', 'day', 'hour']].apply(get_time, axis=1)

#%%
# Add explicitly converter
pd.plotting.register_matplotlib_converters()
# Plt
plt.figure(figsize=(12, 7))
plt.scatter(df5mean['time'], df5mean['pm2.5'])
#%%
df5mean[['pm2.5_shift-1']] = df5mean[['pm2.5']].shift(-1)


#%%

df5mean[['time_shift-1']] = df5mean[['time']].shift(-1)

#%%

df5mean.head()

#%%
# check the next row is the next hour or not. 
# If it is not, the `pm2.5_next_hour` column will be given NaN.

def check_next_hour(x):
    one_hour = dt.timedelta(hours=1)
    if x[2] - x[1] == one_hour:
        return x[0]
    return np.nan

df5mean['pm2.5_next_hour'] = df5mean[['pm2.5_shift-1', 'time', 'time_shift-1']].apply(check_next_hour, axis=1)


#%%

df5mean.head()

#%%

df5mean.isna().sum()

#%%
# Discard rows that contain NaN value
df5mean.dropna(inplace=True)

#%%

df5mean.isna().sum()
#%%
# ### Normalization
# 
# $z = \frac{x- \mu}{\sigma}$
# 

# Save time infomation in another df, and discard it
df5mean_time = df5mean['time_shift-1']
df5mean.drop(columns=['time', 'time_shift-1'], axis=0, inplace=True)
# Normalization
df5mean = (df5mean - df5mean.mean()) / df5mean.std()

#%%
# Divid training set and test set
four_fifth_len = len(df5mean)*0.8
four_fifth_len = int(four_fifth_len)


#%%

train_df = df5mean[:four_fifth_len]
test_df = df5mean[four_fifth_len:]

test_df_time = df5mean_time[four_fifth_len:]

#%%
column = ['month', 'day', 'hour', 'pm2.5', 'pm10.0', 'temp', 'humidity', 'speed']
X = train_df[column]
#X = train_df[['month', 'day', 'weekday', 'hour_minute', 'pm1.0', 'pm2.5', 'pm10.0', 'temp', 'humidity', 'speed']]
y = train_df[['pm2.5_next_hour']]


#%%

test_X = test_df[column]
#test_X = test_df[['month', 'day', 'weekday', 'hour_minute', 'pm1.0', 'pm2.5', 'pm10.0', 'temp', 'humidity', 'speed']]
test_y = test_df[['pm2.5_next_hour']]

#%%
# Fit the model
model = linear_model.LinearRegression(normalize=True)
model.fit(X, y)
#%%
# See the coefficients of our model
a = model.coef_
b = model.intercept_
print(a)
print(b)

#%%

for i in range(len(X.columns)):
    print('Coefficient for %10s:\t%s' % (X.columns[i], model.coef_[0][i]))

#%%
# Calculate mean squared error for training set & test set
predict_train_y = model.predict(X)
predict_y = model.predict(test_X)

train_mse = metrics.mean_squared_error(y, predict_train_y)
test_mse = metrics.mean_squared_error(test_y, predict_y)

print('Train MSE:\t %s' % train_mse)
print('Test MSE:\t %s' % test_mse)


#%%
# Add explicitly converter
pd.plotting.register_matplotlib_converters()
# Plt
plt.figure(figsize=(12, 7))
plt.plot(test_df_time, test_df['pm2.5_next_hour'], label='actual values')
plt.plot(test_df_time, predict_y, label='predict values')
plt.legend()
plt.show()





