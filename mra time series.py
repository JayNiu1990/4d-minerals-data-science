import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cmap
import numpy as np
import pandas as pd
import time
import h5py
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import seaborn as sns
paths = os.listdir('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi MRA Time3\\')
df = []
for i in paths:
    df_sub = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi MRA Time3\\"+str(i))
    df.append(df_sub)
df = pd.concat(df)
df = df[(pd.to_numeric(df["grade"], errors='coerce')>0)]
df = df.reset_index(drop=True)
df = df['grade'].groupby(np.arange(len(df))//450).mean()

# result = adfuller(df.values)
# print('1. ADF:',result[0])
# print('2. p-value:',result[1])
# print('3. Num of Lags:',result[2])
# print('4. Num of Observations Used For ADF Regression and Critical Values Calculation:', result[3])
# print('5. Critical Values:')
# for key,val in result[4].items():
#     print('\t',key,':',val)

# fig, ax = plt.subplots(1,1,figsize=(14,8))
# ax.plot(df)
# ax.set_xlim(0,500)



# import torch
# import torchvision
# from torchvision import datasets
# from torchvision import transforms
# from torch.autograd import Variable
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data.sampler import SubsetRandomSampler

"""
##############ARIMA#######################
data = np.array(df)
p,d,q = 10,1,0
train_size = int(len(data) * 0.7)
train, test = data[0:train_size], data[train_size:len(data)]
history = [x for x in train]

predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(p,d,q))
    model_fit = model.fit()
    pred = model_fit.forecast()
    yhat = pred[0]
    predictions.append(yhat)
    #print(t)
    # Append test observation into overall record
    obs = test[t]
    history.append(obs)
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
plt.plot(test)
plt.plot(predictions,'r')
plt.xlim(0,100)
# plt.ylim(0.5,1.5)
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import time
from scipy.ndimage.interpolation import rotate
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.cm as cmap
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
import os
paths = os.listdir('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi MRA Time3\\')
df = []
for i in paths:
    df_sub = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi MRA Time3\\"+str(i))
    df.append(df_sub)
    

df = pd.concat(df)
df = df[(pd.to_numeric(df["grade"], errors='coerce')>0)]
df = df.reset_index(drop=True)

n=150
df_average_grade = df['grade'].groupby(np.arange(len(df))//n).mean()
df_average_grade = pd.DataFrame(df_average_grade)

date = []
for i in range(len(df_average_grade)):
    date.append(df[150*i+74:150*i+75]['datetime'])
date = pd.concat(date)
date = date.to_frame()
date = date.reset_index(drop=True)

df_average_grade['time'] = date
############switch columns############
columns_titles = ["time","grade"]
df_average_grade=df_average_grade.reindex(columns=columns_titles)

data = df_average_grade['grade']








