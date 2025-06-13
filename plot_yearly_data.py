#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 18:04:20 2025

@author: Fredrik Bergelv
"""


import read_datafiles as read
import csv_data as csv

#%%
import matplotlib.pyplot as plt

string = csv.data["temperature"]["Lund"]
temp = read.get_temp_data(string)
plt.plot(temp['datetime'], temp['temp'])


#%%


read.yearly_histogram(csv.data["temperature"]["Lund"], datatype='temp')