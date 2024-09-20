# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:50:38 2022

@author: niu004
"""
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import plotly.io as pio
import plotly.express as px
############
fields = ['BHID','Fe_dh','As_dh','CuIS_dh','CuT_dh',"X","Y","Z"]
data = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)
data = data[(pd.to_numeric(data["CuT_dh"], errors='coerce')>0) & (pd.to_numeric(data["CuIS_dh"], errors='coerce')>0) & (pd.to_numeric(data["Fe_dh"], errors='coerce')>0) & (pd.to_numeric(data["As_dh"], errors='coerce')>0)]
data["CuT_dh"] = data["CuT_dh"].astype("float")
data["CuIS_dh"] = data["CuIS_dh"].astype("float")
data["Fe_dh"] = data["Fe_dh"].astype("float")
data["As_dh"] = data["As_dh"].astype("float")
data["Fe_dh_log"] = np.log10(data["Fe_dh"])
data["As_dh_log"] = np.log10(data["As_dh"])
data["CuT_dh_log"] = np.log10(data["CuT_dh"])
data["CuIS_dh_log"] = np.log10(data["CuIS_dh"])
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import collections
each_name = [item for item, count in collections.Counter(data["BHID"]).items() if count > 5]
index_pos = [list(data["BHID"]).index(stri) for stri in each_name]
list1 = [data[i:i+1] for i in index_pos]
list1 = pd.concat(list1)
data1 = [data[data["BHID"] =="%s"%(stri)] for stri in each_name]

    
from sklearn.linear_model import LinearRegression

r_squared_list = []
for i in range(len(data1)):
    model = LinearRegression()
    x,y = np.array(data1[i]["Fe_dh_log"]), np.array(data1[i]["CuT_dh_log"])
    model.fit(x.reshape(-1,1), y)
    r_squared = model.score(x.reshape(-1,1), y)
    if r_squared >0:
        r_squared_list.append(r_squared)
    else:
        r_squared_list.append(0)
list1["R2_Fe_log"] = r_squared_list

list1_high_r2 = list1[list1["R2_Fe_log"]>=0.3]
list1_median_r2 = list1[(list1["R2_Fe_log"]<0.3) & (list1["R2_Fe_log"]>=0.1)]
list1_low_r2 = list1[list1["R2_Fe_log"]<0.1]
plt.scatter(list(list1_high_r2['X']),list(list1_high_r2['Y']))
plt.scatter(list(list1_median_r2['X']),list(list1_median_r2['Y']))
plt.scatter(list(list1_low_r2['X']),list(list1_low_r2['Y']))
import plotly.express as px
data1_low_copper = []
data1_median_copper = []
data1_high_copper = []

for i in range(len(data1)):
    if data1[i]["CuT_dh"].mean() <= 0.1:
        data1_low_copper.append(data1[i])
    elif data1[i]["CuT_dh"].mean() > 0.1 and data1[i]["CuT_dh"].mean() < 0.5:
        data1_median_copper.append(data1[i])
    elif data1[i]["CuT_dh"].mean() >= 0.5:
        data1_high_copper.append(data1[i])
    else:
        pass
data1_low_copper = pd.concat(data1_low_copper)
data1_median_copper = pd.concat(data1_median_copper)
data1_high_copper = pd.concat(data1_high_copper)

data1_low_copper["label"] = pd.Series("<0.1",index=data1_low_copper.index)
data1_median_copper["label"] = pd.Series("0.1-1",index=data1_median_copper.index)
data1_high_copper["label"] = pd.Series(">1",index=data1_high_copper.index)

data1_total = pd.concat([data1_low_copper,data1_median_copper,data1_high_copper])

fig = px.scatter_3d(data1_total, x="X",y="Y",z="Z",color = "label")
fig.update_traces(marker_size=3)
fig.show()

average_grade =[]
for i in range(len(data1)):
    average_grade.append(data1[i]["CuT_dh"].mean())
    


list1["Average_copper"] = average_grade
list1["Average_copper_label"] = pd.cut(list1["Average_copper"],[0,0.1,0.3,0.5,1,100],labels=["<0.1","0.1-0.3","0.3-0.5","0.5-1",">1"])
fig = px.scatter(list1,x = "X", y = "Y", color = "Average_copper_label")
fig.show()


import plotly.io as pio
pio.renderers.default = "browser"
list1["R2_Fe_label"] = pd.cut(list1["R2_Fe_log"],[0,0.1,0.2,0.5,0.7,1],labels=["<0.1","0.1-0.2","0.2-0.5","0.5-0.7",">0.7"])
list1_1 = list1[list1["R2_Fe_label"].str.contains("nan")==False]
fig = px.scatter(list1_1,x = "X", y = "Y", color = "R2_Fe_label")
fig.show()

import plotly.io as pio
pio.renderers.default = "browser"

r_squared_list1 = []
for i in range(len(data1)):
    model = LinearRegression()
    x,y = np.array(data1[i]["As_dh"]), np.array(data1[i]["CuT_dh"])
    model.fit(x.reshape(-1,1), y)
    r_squared = model.score(x.reshape(-1,1), y)
    if r_squared >0:
        r_squared_list1.append(r_squared)
    else:
        r_squared_list1.append(0)
list1["R2_As"] = r_squared_list1
list1["R2_As_label"] = pd.cut(list1["R2_As"],[0,0.1,0.2,0.5,0.7,1],labels=["<0.1","0.1-0.2","0.2-0.5","0.5-0.7",">0.7"])
list1_2 = list1[list1["R2_As_label"].str.contains("nan")==False]
fig = px.scatter(list1_2,x = "X", y = "Y", color = "R2_As_label")
fig.show()


















