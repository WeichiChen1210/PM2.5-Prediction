#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 23:45:37 2019

@author: weichi
"""
import time as t
import datetime as dt
import meteor_data_crawler as weather
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

#%% get wind speed and direction data without lost days
wind_data_list = []
start = t.time()

for month in range(6, 9):
    max_day = 32
    if month == 6:
        max_day = 31
    if month == 8:
        max_day = 15
    month_str = str(month)
    for day in range(1, max_day):
        wind_day = weather.crawler(month, day)
        for hour in wind_day:
            wind_data_list.append(hour)

    print("Finish "+ str(month))

end = t.time()
print(end-start)

# rename colomn names
title = ['month', 'day', 'hour', 'speed']
df_wind = pd.DataFrame(data=wind_data_list, columns=title)
#%% save original data
df_wind.to_csv('complete_wind.csv')
#%%
# Input time
time_interval = ['2019 06 01', '2019 08 15']
taipei_tz = pytz.timezone('Asia/Taipei')

# Set time
start_time = dt.datetime.strptime(time_interval[0], '%Y %m %d').replace(tzinfo=taipei_tz)
end_time = dt.datetime.strptime(time_interval[1], '%Y %m %d').replace(tzinfo=taipei_tz)

#%% get data
pos5 = get_data_by_pos(5)
df5 = pd.DataFrame(pos5)
#pos6 = get_data_by_pos(6)
#df6 = pd.DataFrame(pos6)
#pos7 = get_data_by_pos(7)
#df7 = pd.DataFrame(pos7)

# Select the duration
df5 = df5.loc[df5['date'] >= start_time]
df5 = df5.loc[df5['date'] <= end_time]
df5 = df5[1:]
# Rename the names of columns
df5 = df5.rename(columns = {'pm10': 'pm1.0', 'pm25': 'pm2.5', 'pm100': 'pm10.0'})

# Exclude outliers
want_cols = ['humidity', 'pm1.0', 'pm10.0', 'pm2.5', 'temp']
df5 = df5[(np.abs(stats.zscore(df5.loc[:, want_cols])) < 4).all(axis=1)]

#df6 = df6.loc[df6['date'] >= start_time]
#df6 = df6.loc[df6['date'] <= end_time]
#df6 = df6[1:]
#df6 = df6.rename(columns = {'pm10': 'pm1.0', 'pm25': 'pm2.5', 'pm100': 'pm10.0'})
#df6 = df6[(np.abs(stats.zscore(df6.loc[:, want_cols])) < 4).all(axis=1)]
#
#df7 = df7.loc[df7['date'] >= start_time]
#df7 = df7.loc[df7['date'] <= end_time]
#df7 = df7[1:]
#df7 = df7.rename(columns = {'pm10': 'pm1.0', 'pm25': 'pm2.5', 'pm100': 'pm10.0'})
#df7 = df7[(np.abs(stats.zscore(df7.loc[:, want_cols])) < 4).all(axis=1)]

#%% Split time infomation from column `date`
column = ['month', 'day', 'hour', 'pm1.0', 'pm2.5', 'pm10.0', 'temp', 'humidity']
df5['month'] = df5['date'].apply(lambda x: x.month)
df5['day'] = df5['date'].apply(lambda x: x.day)
df5['hour'] = df5['date'].apply(lambda x: x.hour)
df5 = df5[column]

# df5['hour_minute'] = df5['date'].apply(lambda x: x.hour+x.minute/60)
# df5['weekday'] = df5['date'].apply(lambda x: x.weekday)

# df5 = df5[['month', 'day', 'weekday', 'hour', 'hour_minute', 'pm1.0', 'pm2.5', 'pm10.0', 'temp', 'humidity']]

#df6['month'] = df6['date'].apply(lambda x: x.month)
#df6['day'] = df6['date'].apply(lambda x: x.day)
#df6['hour'] = df6['date'].apply(lambda x: x.hour)
#df6 = df6[column]
#
#df7['month'] = df7['date'].apply(lambda x: x.month)
#df7['day'] = df7['date'].apply(lambda x: x.day)
#df7['hour'] = df7['date'].apply(lambda x: x.hour)
#df7 = df7[column]

#%% Evaluate mean values for each hour
df5mean = df5.groupby(['month', 'day', 'hour']).mean()
df5mean.reset_index(inplace=True)

#df6mean = df6.groupby(['month', 'day', 'hour']).mean()
#df6mean.reset_index(inplace=True)
#
#df7mean = df7.groupby(['month', 'day', 'hour']).mean()
#df7mean.reset_index(inplace=True)
#%%
df5mean.to_csv('pos5.csv')
#df6mean.to_csv('pos6.csv')
#df7mean.to_csv('pos7.csv')

#%% Create a new dataframe with complete hours
num = (30 + 31 + 14) * 24
# cols = ['month', 'day', 'weekday', 'hour', 'hour_minute', 'pm1.0', 'pm2.5', 'pm10.0', 'temp', 'humidity']
cols = ['month', 'day', 'hour', 'pm1.0', 'pm2.5', 'pm10.0', 'temp', 'humidity']
complete_5 = pd.DataFrame(columns = cols, index=range(0, num))
count_hour = 0
count_day = 1
count_month = 6
for index, rows in complete_5.iterrows():
    rows['hour'] = count_hour
    rows['day'] = count_day
    rows['month'] = count_month
    if count_hour == 23:
        count_hour = 0
        count_day += 1
        if count_day > 30:
            if count_month == 6:
                count_month = 7
                count_day = 1
        if count_day > 31:
            count_month += 1
            count_day = 1           
    else:
        count_hour += 1

#complete_6 = complete_5.copy()
#complete_7 = complete_5.copy()

#%% Fill the existing data to a new and complete one
# can do it with an easier way
start = t.time()

for index, rows in df5mean.iterrows():
    month = rows['month']
    day = rows['day']
    hour = rows['hour']
    # new.loc[(new['month'] == month) & (new['day'] == day) & (new['hour'] == hour), 'weekday'] = rows['weekday']
    # new.loc[(new['month'] == month) & (new['day'] == day) & (new['hour'] == hour), 'hour_minute'] = rows['hour_minute']
    complete_5.loc[(complete_5['month'] == month) & (complete_5['day'] == day) & (complete_5['hour'] == hour), 'pm1.0'] = rows['pm1.0']
    complete_5.loc[(complete_5['month'] == month) & (complete_5['day'] == day) & (complete_5['hour'] == hour), 'pm2.5'] = rows['pm2.5']
    complete_5.loc[(complete_5['month'] == month) & (complete_5['day'] == day) & (complete_5['hour'] == hour), 'pm10.0'] = rows['pm10.0']
    complete_5.loc[(complete_5['month'] == month) & (complete_5['day'] == day) & (complete_5['hour'] == hour), 'temp'] = rows['temp']
    complete_5.loc[(complete_5['month'] == month) & (complete_5['day'] == day) & (complete_5['hour'] == hour), 'humidity'] = rows['humidity']

end = t.time()
print(end-start)
#%% for interpolation
complete_5 = complete_5.dropna()
complete_5 = complete_5.reset_index()
complete_5 = complete_5.set_index(complete_5.pop('index'))
complete_5 = complete_5.reindex(np.arange(complete_5.index.min(), complete_5.index.max()+1))
for col in complete_5:
    complete_5[col] = pd.to_numeric(complete_5[col], errors='coerce')
#%% interpolation
complete_5 = complete_5.interpolate()
#%% correcting dates
hour = 19
day = 6
for i in range(139, 209):
    complete_5.iloc[i, 1] = day
    complete_5.iloc[i, 2] = hour
    hour += 1
    if hour > 23:
        hour = 0
        day += 1
complete_5.iloc[432:434, 1] = 19
complete_5.iloc[432, 2] = 0
complete_5.iloc[433, 2] = 1
#%%
complete_5 = complete_5.astype({'month': int, 'day': int, 'hour': int})
decimals = pd.Series([3, 3, 3, 3, 3], index=['pm1.0', 'pm2.5', 'pm10.0', 'temp', 'humidity'])
complete_5 = complete_5.round(decimals)

#%%
df_wind = df_wind.drop(['month', 'day', 'hour'], axis=1)
df_wind = df_wind.round(3)
# concate the two frames
complete_5 = pd.concat([complete_5, df_wind], axis=1, sort=False)

#%%
complete_5.to_csv('complete_data_5.csv')
#%%
#for index, rows in df6mean.iterrows():
#    month = rows['month']
#    day = rows['day']
#    hour = rows['hour']
#    # new.loc[(new['month'] == month) & (new['day'] == day) & (new['hour'] == hour), 'weekday'] = rows['weekday']
#    # new.loc[(new['month'] == month) & (new['day'] == day) & (new['hour'] == hour), 'hour_minute'] = rows['hour_minute']
#    complete_6.loc[(complete_6['month'] == month) & (complete_6['day'] == day) & (complete_6['hour'] == hour), 'pm1.0'] = rows['pm1.0']
#    complete_6.loc[(complete_6['month'] == month) & (complete_6['day'] == day) & (complete_6['hour'] == hour), 'pm2.5'] = rows['pm2.5']
#    complete_6.loc[(complete_6['month'] == month) & (complete_6['day'] == day) & (complete_6['hour'] == hour), 'pm10.0'] = rows['pm10.0']
#    complete_6.loc[(complete_6['month'] == month) & (complete_6['day'] == day) & (complete_6['hour'] == hour), 'temp'] = rows['temp']
#    complete_6.loc[(complete_6['month'] == month) & (complete_6['day'] == day) & (complete_6['hour'] == hour), 'humidity'] = rows['humidity']
#
#for index, rows in df7mean.iterrows():
#    month = rows['month']
#    day = rows['day']
#    hour = rows['hour']
#    # new.loc[(new['month'] == month) & (new['day'] == day) & (new['hour'] == hour), 'weekday'] = rows['weekday']
#    # new.loc[(new['month'] == month) & (new['day'] == day) & (new['hour'] == hour), 'hour_minute'] = rows['hour_minute']
#    complete_7.loc[(complete_7['month'] == month) & (complete_7['day'] == day) & (complete_7['hour'] == hour), 'pm1.0'] = rows['pm1.0']
#    complete_7.loc[(complete_7['month'] == month) & (complete_7['day'] == day) & (complete_7['hour'] == hour), 'pm2.5'] = rows['pm2.5']
#    complete_7.loc[(complete_7['month'] == month) & (complete_7['day'] == day) & (complete_7['hour'] == hour), 'pm10.0'] = rows['pm10.0']
#    complete_7.loc[(complete_7['month'] == month) & (complete_7['day'] == day) & (complete_7['hour'] == hour), 'temp'] = rows['temp']
#    complete_7.loc[(complete_7['month'] == month) & (complete_7['day'] == day) & (complete_7['hour'] == hour), 'humidity'] = rows['humidity']









