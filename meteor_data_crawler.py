#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 23:44:39 2019

@author: weichi
"""
import requests as rq
from bs4 import BeautifulSoup
import numpy as np

station = str(467410)
pre_url = 'https://e-service.cwb.gov.tw/HistoryDataQuery/DayDataController.do?command=viewMain&station=' + station + '&stname=&datepicker='
#%%
def rain_wind_crawler(month, date):
    # create url    
    mon = ''
    if month < 10:
        mon = '0' + str(month)
    else:
        mon = str(month)
    
    day = ''
    if date < 10:
        day = '0' + str(date)
    else:
        day = str(date)
    datepicker = '2019-' + mon + '-' + day
    
    # url: https://e-service.cwb.gov.tw/HistoryDataQuery/DayDataController.do?command=viewMain&station=467410&stname=&datepicker=2019-08-07
    url = pre_url + datepicker
    # print(url)
    
    # request
    response = rq.get(url)
    # print(response.text)
    
    # html parsing
    soup = BeautifulSoup(response.text, features="html.parser")
    
#    title = ['WS', 'WD']
    
    # get the daily data
    body = soup.tbody
    trs = body.find_all('tr')
    trs = trs[3:]
    
    data = []
    hour = 0
    # extract wind speed and wind direction
    for tds in trs:
        sd = {}
        td = tds.find_all('td')
        
        if td[7].string == "V\xa0":            
            sd['wd'] = np.nan
        else:            
            sd['wd'] = float(td[7].string)
        
        if td[10].string == "T\xa0":
            sd['precp'] = float(0.05)
        else:
            sd['precp'] = float(td[10].string)
            
        sd['month'] = month
        sd['day'] = date
        sd['hour'] = hour
        sd['ws'] = float(td[6].string)        
        
        data.append(sd)
        hour += 1
    
    # turn the list to dataframe
    #df = pd.DataFrame(data=winddata, columns=title)
    
    return data
  
    