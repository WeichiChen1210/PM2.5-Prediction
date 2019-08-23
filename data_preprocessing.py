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

#%% get wind speed, direction and precipitation data
data_list = []
start = t.time()

for month in range(6, 9):
    max_day = 32
    if month == 6:
        max_day = 31
    if month == 8:
        max_day = 22
    month_str = str(month)
    for day in range(1, max_day):
        data = weather.rain_wind_crawler(month, day)
        for hour in data:
            data_list.append(hour)

    print("Finish "+ str(month))

end = t.time()
print(end-start)

# rename colomn names
title = ['month', 'day', 'hour', 'ws']
df_wspd = pd.DataFrame(data=data_list, columns=title)
title = ['month', 'day', 'hour', 'wd']
df_wdir = pd.DataFrame(data=data_list, columns=title)
title = ['month', 'day', 'hour', 'precp']
df_precp = pd.DataFrame(data=data_list, columns=title)

#%% save original data
df_wspd.to_csv('complete_wind_speed.csv')
df_wdir.to_csv('complete_wind_direction.csv')
df_precp.to_csv('complete_precp.csv')

#%% read from csv files
df_wspd = pd.read_csv('complete_wind_speed.csv', index_col=0)
df_wdir = pd.read_csv('complete_wind_direction.csv', index_col=0)
df_precp = pd.read_csv('complete_precp.csv', index_col=0)
#%%
# Input time
time_interval = ['2019 06 01', '2019 08 22']
taipei_tz = pytz.timezone('Asia/Taipei')

# Set time
start_time = dt.datetime.strptime(time_interval[0], '%Y %m %d').replace(tzinfo=taipei_tz)
end_time = dt.datetime.strptime(time_interval[1], '%Y %m %d').replace(tzinfo=taipei_tz)

#%% get data
pos5 = get_data_by_pos(5)
df5 = pd.DataFrame(pos5)

# Select the duration
df5 = df5.loc[df5['date'] >= start_time]
df5 = df5.loc[df5['date'] <= end_time]
df5 = df5[1:]
# Rename the names of columns
df5 = df5.rename(columns = {'pm10': 'pm1.0', 'pm25': 'pm2.5', 'pm100': 'pm10.0'})

# Exclude outliers
df5 = df5.loc[ df5['pm2.5'] <= 120 ]
df5 = df5.loc[ df5['humidity'] <= 100 ]

#%% Split time infomation from column `date`
column = ['month', 'day', 'hour', 'pm1.0', 'pm2.5', 'pm10.0', 'temp', 'humidity']
df5['month'] = df5['date'].apply(lambda x: x.month)
df5['day'] = df5['date'].apply(lambda x: x.day)
df5['hour'] = df5['date'].apply(lambda x: x.hour)
df5 = df5[column]

#%% Evaluate mean values for each hour
df5mean = df5.groupby(['month', 'day', 'hour']).mean()
df5mean.reset_index(inplace=True)

#%%
df5mean.to_csv('pos5.csv')

#%%
df5mean = pd.read_csv('pos5.csv', index_col=0)

#%% Create a new dataframe with complete hours
num = (30 + 31 + 21) * 24
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

#%% Fill the existing data to a new and complete one
# can do it with an easier way
start = t.time()

for index, rows in df5mean.iterrows():
    month = rows['month']
    day = rows['day']
    hour = rows['hour']
    s = pd.Series({'month': month, 'day': day, 'hour': hour, 'pm1.0': rows['pm1.0'], 'pm2.5': rows['pm2.5'], 'pm10.0': rows['pm10.0'], 'temp': rows['temp'], 'humidity': rows['humidity']})
    complete_5.loc[(complete_5['month'] == month) & (complete_5['day'] == day) & (complete_5['hour'] == hour), :] = s[complete_5.columns].values

end = t.time()
print(end-start)

#%% (no interpolation) drop time cols and concate with complete_5
drop_wspd = df_wspd.drop(['month', 'day', 'hour'], axis=1)
drop_wdir = df_wdir.drop(['month', 'day', 'hour'], axis=1)
drop_precp = df_precp.drop(['month', 'day', 'hour'], axis=1)

complete_5 = pd.concat([complete_5, drop_wspd], axis=1, sort=False)
complete_5 = pd.concat([complete_5, drop_wdir], axis=1, sort=False)
complete_5 = pd.concat([complete_5, drop_precp], axis=1, sort=False)

#%% (no interpolation) drop nan and change data type
complete_5 = complete_5.dropna()
complete_5 = complete_5.reset_index(drop=True)
complete_5 = complete_5.astype({'month': int, 'day': int, 'hour': int, 'pm1.0': float, 'pm2.5': float, 'pm10.0': float, 'temp': float, 'humidity': float})

# jump to the last section-save file

#%% for interpolation
complete_5 = complete_5.dropna()
complete_5 = complete_5.reset_index()
complete_5 = complete_5.set_index(complete_5.pop('index'))
complete_5 = complete_5.reindex(np.arange(complete_5.index.min(), complete_5.index.max()+1))
for col in complete_5:
    complete_5[col] = pd.to_numeric(complete_5[col], errors='coerce')

#%% correcting dates
hour = 19
day = 6
month = 6
for i in range(139, 209):
    for j in range(3, 8):
        complete_5.iloc[i, j] = (complete_5.iloc[i+72, j]+complete_5.iloc[i-72, j])/2 
    complete_5.iloc[i, 0] = month
    complete_5.iloc[i, 1] = day
    complete_5.iloc[i, 2] = hour
    hour += 1
    if hour > 23:
        hour = 0
        day += 1
complete_5.iloc[432:448, 0] = 6
complete_5.iloc[432:448, 1] = 19
complete_5.iloc[432, 2] = 0
complete_5.iloc[433, 2] = 1
complete_5.iloc[436, 2] = 4
complete_5.iloc[447, 2] = 15

#%% interpolation
complete_5 = complete_5.interpolate()

#%%
complete_5 = complete_5.astype({'month': int, 'day': int, 'hour': int})
decimals = pd.Series([3, 3, 3, 3, 3], index=['pm1.0', 'pm2.5', 'pm10.0', 'temp', 'humidity'])
complete_5 = complete_5.round(decimals)

#%%
df_wspd = df_wspd.drop(['month', 'day', 'hour'], axis=1)
df_wspd = df_wspd.round(3)
# concate the two frames
complete_5 = pd.concat([complete_5, df_wspd], axis=1, sort=False)

df_precp = df_precp.drop(['month', 'day', 'hour'], axis=1)
df_precp = df_precp.round(3)
# concate the two frames
complete_5 = pd.concat([complete_5, df_precp], axis=1, sort=False)


#%% save the file
complete_5.to_csv('complete_data_5.csv')






