#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 22:57:54 2019

@author: weichi
"""
import os
import datetime as dt
import wind_data_crawler as wind
import requests
import json
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics, model_selection
from scipy import stats

def get_data_by_pos(pos):
    r = requests.get(f'http://140.116.82.93:6800/campus/display/{ pos }')
    # date field in self.data is the str of datetime
    # We need to convert it to timezone aware object first
    data = json.loads(r.text)
    for index, value in enumerate(data):
      # strptime() parse str of date according to the format given behind
      # It is still naive datetime object, meaning that it is unaware of timezone
      unaware = dt.datetime.strptime(value.get('date'),  '%a, %d %b %Y %H:%M:%S %Z')
      # Create a utc timezone
      utc_timezone = pytz.timezone('UTC')
      # make utc_unaware obj aware of timezone
      # Convert the given time directly to literally the same time with different timezone
      # For example: Change from 2019-05-19 07:41:13(unaware) to 2019-05-19 07:41:13+00:00(aware, tzinfo=UTC)
      utc_aware = utc_timezone.localize(unaware)
      # This can also do the same thing
      # Replace the tzinfo of an unaware datetime object to a given tzinfo
      # utc_aware = unaware.replace(tzinfo=pytz.utc)

      # Transform utc timezone to +8 GMT timezone
      # Convert the given time to the same moment of time just like performing timezone calculation
      # For example: Change from 2019-05-19 07:41:13+00:00(aware, tzinfo=UTC) to 2019-05-19 15:41:13+08:00(aware, tzinfo=Asiz/Taipei)
      taiwan_aware = utc_aware.astimezone(pytz.timezone('Asia/Taipei'))
      # print(f"{ index }: {unaware} {utc_aware} {taiwan_aware}")
      value['date'] = taiwan_aware
    return data

def get_all_data():
    r = requests.get(f'http://140.116.82.93:6800/training')
    # date field in self.data is the str of datetime
    # We need to convert it to timezone aware object first
    data = json.loads(r.text)
    for index, value in enumerate(data):
      # strptime() parse str of date according to the format given behind
      # It is still naive datetime object, meaning that it is unaware of timezone
      unaware = dt.datetime.strptime(value.get('date'),  '%a, %d %b %Y %H:%M:%S %Z')
      # Create a utc timezone
      utc_timezone = pytz.timezone('UTC')
      # make utc_unaware obj aware of timezone
      # Convert the given time directly to literally the same time with different timezone
      # For example: Change from 2019-05-19 07:41:13(unaware) to 2019-05-19 07:41:13+00:00(aware, tzinfo=UTC)
      utc_aware = utc_timezone.localize(unaware)
      # This can also do the same thing
      # Replace the tzinfo of an unaware datetime object to a given tzinfo
      # utc_aware = unaware.replace(tzinfo=pytz.utc)

      # Transform utc timezone to +8 GMT timezone
      # Convert the given time to the same moment of time just like performing timezone calculation
      # For example: Change from 2019-05-19 07:41:13+00:00(aware, tzinfo=UTC) to 2019-05-19 15:41:13+08:00(aware, tzinfo=Asiz/Taipei)
      taiwan_aware = utc_aware.astimezone(pytz.timezone('Asia/Taipei'))
      # print(f"{ index }: {unaware} {utc_aware} {taiwan_aware}")
      value['date'] = taiwan_aware
    return data

# Reconstruct time infomation by `month`, `day`, and `hour`

def get_time(x):
    time_str = '2019 %d %d %d' % (x[0], x[1], x[2])
    taipei_tz = pytz.timezone('Asia/Taipei')
    time = dt.datetime.strptime(time_str, '%Y %m %d %H').replace(tzinfo=taipei_tz)
    return time

#today = int(dt.datetime.now().strftime("%d"))

#%% get data
pos5 = get_data_by_pos(5)
pos3 = get_data_by_pos(3)
df5 = pd.DataFrame(pos5)
df3 = pd.DataFrame(pos3)
#%%
# Input time
time = ['2019 06 01', '2019 08 10']
taipei_tz = pytz.timezone('Asia/Taipei')

# Set time
start_time = dt.datetime.strptime(time[0], '%Y %m %d').replace(tzinfo=taipei_tz)
end_time = dt.datetime.strptime(time[1], '%Y %m %d').replace(tzinfo=taipei_tz)

# Select the duration
df5 = df5.loc[df5['date'] >= start_time]
df5 = df5.loc[df5['date'] <= end_time]
df5 = df5[1:]
#%%
# Rename the names of columns
df5 = df5.rename(columns = {'pm10': 'pm1.0', 'pm25': 'pm2.5', 'pm100': 'pm10.0'})

# Exclude outliers
want_cols = ['humidity', 'pm1.0', 'pm10.0', 'pm2.5', 'temp']
df5 = df5[(np.abs(stats.zscore(df5.loc[:, want_cols])) < 4).all(axis=1)]

#%% 
# Split time infomation from column `date`
df5['month'] = df5['date'].apply(lambda x: x.month)
df5['day'] = df5['date'].apply(lambda x: x.day)
df5['weekday'] = df5['date'].apply(lambda x: x.weekday)
df5['hour'] = df5['date'].apply(lambda x: x.hour)
df5['hour_minute'] = df5['date'].apply(lambda x: x.hour+x.minute/60)

#%%
df5 = df5[['month', 'day', 'weekday', 'hour', 'hour_minute', 'pm1.0', 'pm2.5', 'pm10.0', 'temp', 'humidity']]

#%%
# Evaluate mean values for each hour
df5mean = df5.groupby(['month', 'day', 'hour']).mean()
#%%
df5mean.reset_index(inplace=True)

df5mean.to_csv('pos5.csv')
#%% Find the lost days
count = 1
count_hour = 0
lost_day = {'6': [], '7': [], '8': []}
temp_list = []
for index, rows in df5mean.iterrows():
    month = rows['month']
    day = rows['day']
    hour = rows['hour']
    if day != count:        
        if day == (count + 1):
            count = day
        elif day < count:            
            # print(day, count)
            count = 1
            while count != day:
                temp_list.append(count)
                count += 1
            tmp = str(int(month))
            lost_day[tmp] += temp_list
            temp_list.clear()
            count = day
        else:
            # print(day, count)
            count += 1
            while count != day:
                temp_list.append(count)
                count += 1
            tmp = str(int(month))
            lost_day[tmp] += temp_list
            temp_list.clear()
            count = day     
        
    if hour != count_hour:
        
        print(month, day, hour)
        count_hour = hour + 1
    else:
        count_hour += 1
    if count_hour > 23:
        count_hour = 0
#%% get wind speed and direction data without lost days
wind_data_list = []

for month in range(6, 9):
    max_day = 32
    if month == 6:
        max_day = 31
    if month == 8:
        max_day = 10
    month_str = str(month)
    for day in range(1, max_day):
        # if the day is missing, ignore
        if day in lost_day[month_str]:
            continue        
        wind_day = wind.crawler(month, day)
        for hour in wind_day:
            wind_data_list.append(hour)

    print("Finish "+ str(month))

#%% rename colomn names
title = ['month', 'day', 'hour', 'speed']
df_org_wind = pd.DataFrame(data=wind_data_list, columns=title)
#%% save original data and make a copy
df_org_wind.to_csv('complete_wind.csv')
df_wind = df_org_wind.copy()

#%% delete extra data
for i in range(139, 161):
    df_wind = df_wind.drop(i)

del_list = [384, 385, 388, 399]
for i in del_list:
    df_wind = df_wind.drop(i)
#%% reset index of wind data and save to csv
df_wind.reset_index(inplace=True, drop=True)

df_wind.to_csv('wind.csv')

#%% read data from saved file
df5mean = pd.read_csv('pos5.csv', index_col=0)

df_wind = pd.read_csv('wind.csv', index_col=0)
df_wind = df_wind.drop(['month', 'day', 'hour'], axis=1)
#%%
# concate the two frames
df5mean = pd.concat([df5mean, df_wind], axis=1, sort=False)

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


# ### Normalization
# 
# $z = \frac{x- \mu}{\sigma}$
# 

#%%

# Save time infomation in another df, and discard it
df5mean_time = df5mean['time_shift-1']
df5mean.drop(columns=['time', 'time_shift-1'], axis=0, inplace=True)
# Normalization
df5mean = (df5mean - df5mean.mean()) / df5mean.std()


# ### Divid training set and test set

#%%

four_fifth_len = len(df5mean)*0.8
four_fifth_len = int(four_fifth_len)


#%%

train_df = df5mean[:four_fifth_len]
test_df = df5mean[four_fifth_len:]

test_df_time = df5mean_time[four_fifth_len:]


#%%

#X = train_df[['month', 'hour_minute', 'pm1.0', 'pm2.5', 'pm10.0', 'temp', 'humidity', 'speed']]
X = train_df[['month', 'day', 'weekday', 'hour_minute', 'pm1.0', 'pm2.5', 'pm10.0', 'temp', 'humidity', 'speed']]
y = train_df[['pm2.5_next_hour']]


#%%

#test_X = test_df[['month', 'hour_minute', 'pm1.0', 'pm2.5', 'pm10.0', 'temp', 'humidity', 'speed']]
test_X = test_df[['month', 'day', 'weekday', 'hour_minute', 'pm1.0', 'pm2.5', 'pm10.0', 'temp', 'humidity', 'speed']]
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





