#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 22:57:54 2019

@author: weichi
"""
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


today = int(dt.datetime.now().strftime("%d"))

#%%
pos5 = get_data_by_pos(5)

#%%
df5 = pd.DataFrame(pos5)

# Input time
time = ['2019 06 01', '2019 08 08']
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

#%%
max_day = 30
count = 1
lost_day = []
for index, rows in df5mean.iterrows():
    if rows['day'] != count:        
        if rows['day'] == (count + 1):
            count = rows['day']
        elif rows['day'] < count:            
            print(rows['day'], count)
            count = 1
            while count != rows['day']:
                lost_day.append(count)
                count += 1
            count = rows['day']
        else:
            print(rows['day'], count)
            while count != rows['day']:
                lost_day.append(count)
                count += 1
            count = rows['day']       
        
        
    
#%% get wind speed and direction data
wind_data_list = []

for month in range(6, 9):
    max_day = 32
    if month == 6:
        max_day = 31
    if month == 8:
        # max_day = today-1
        max_day = 8
    for day in range(1, max_day):        
        wind_day = wind.crawler(month, day)
        for hour in wind_day:
            wind_data_list.append(hour)

    print("Finish "+ str(month))

#%%
title = ['month', 'day', 'hour', 'speed']
df_wind = pd.DataFrame(data=wind_data_list, columns=title)
#%%
# Reconstruct time infomation by `month`, `day`, and `hour`

def get_time(x):
    time_str = '2019 %d %d %d' % (x[0], x[1], x[2])
    taipei_tz = pytz.timezone('Asia/Taipei')
    time = dt.datetime.strptime(time_str, '%Y %m %d %H').replace(tzinfo=taipei_tz)
    return time

df5mean['time'] = df5mean[['month', 'day', 'hour']].apply(get_time, axis=1)

















