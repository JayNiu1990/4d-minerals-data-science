# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 16:06:14 2022

@author: niu004
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_digits, load_diabetes, make_regression
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import collections
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'
fields = ['BHID', 'X','Y','Z','CuT_dh','Fe_dh','LITH','As_dh']
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Escondida Trucks\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)
df = df.loc[(pd.to_numeric(df["CuT_dh"], errors='coerce')>0.02) &(pd.to_numeric(df["Fe_dh"], errors='coerce')>0)
            &(pd.to_numeric(df["As_dh"], errors='coerce')>0) &(pd.to_numeric(df["LITH"], errors='coerce')>0)]
df['LITH'] = df['LITH'].astype(int)
df["CuT_dh"] = df["CuT_dh"].astype("float")
df["Fe_dh"] = df["Fe_dh"].astype("float")
df["As_dh"] = df["As_dh"].astype("float")
each_name = [item for item, count in collections.Counter(df["LITH"]).items() if count > 100]
df = df.loc[(df['LITH']==each_name[0])]
stdsc = StandardScaler()
X = stdsc.fit_transform(np.array(df['X']).reshape(-1,1))
Y = stdsc.fit_transform(np.array(df['Y']).reshape(-1,1))
Z = stdsc.fit_transform(np.array(df['Z']).reshape(-1,1))
CuT = np.log(np.array(df['CuT_dh'])).reshape(-1,1).reshape(-1)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingRegressor
from sklearn import neighbors
from skelm import ELMRegressor
inputdata = np.concatenate((X,Y,Z),axis=1)
x_train, x_test, y_train, y_test = train_test_split(inputdata, CuT, test_size=0.2,random_state=10)










# estimators = [('random_forest',RandomForestRegressor(n_estimators=150)),
#               ('KNN',neighbors.KNeighborsRegressor(weights='distance',n_neighbors=5,leaf_size=50,algorithm='kd_tree')),
#               ('ELM',ELMRegressor(ufunc='relu',n_neurons=100,alpha=0.001))
#               ]
# from sklearn.ensemble import StackingRegressor,GradientBoostingRegressor
# final_estimator = GradientBoostingRegressor(n_estimators=25,subsample=0.5,min_samples_leaf=25,max_features=1)
# reg = StackingRegressor(estimators=estimators,final_estimator=final_estimator)
# reg.fit(x_train,y_train)
# y_pred = reg.predict(x_test)
# plt.scatter(y_pred,y_test)