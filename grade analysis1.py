# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 17:03:32 2022

@author: niu004
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import glob
import os
data1 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\Kansanshi\\Trial Data - Copy\\2min_6min_time_elapse_individual_truck_grade_Oct21_Mar22.csv")
df = data1.mean(axis = 1)
plt.hist(df,bins=200)
plt.show()
pd.DataFrame(df).plot(kind='density') 
