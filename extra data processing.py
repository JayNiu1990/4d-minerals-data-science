import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd    
import plotly.io as pio
import plotly.graph_objs as go
import arviz as az
from scipy import stats
import random
pio.renderers.default='browser'
import os
items = os.listdir('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi MRA Time extra data\\')
for i in items:
    df = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi MRA Time extra data\\'+str(i))
    df['hour'] = pd.to_datetime(df['TIME']).dt.hour
    df['minute'] = pd.to_datetime(df['TIME']).dt.minute
    df['second'] = pd.to_datetime(df['TIME']).dt.second
    df['tonnage'] = df['TPH']
    df['grade'] = df['GRADE']
    df = df.drop(columns=['TIME','TPH','GRADE'])
    df.to_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi MRA Time extra data\\new-'+str(i),header=False,index=False)