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

n=1
df_average_grade = df['grade'].groupby(np.arange(len(df))//n).mean()
df_average_grade = pd.DataFrame(df_average_grade)[0:2000]



data = df_average_grade['grade'].values
from statsmodels.tsa.stattools import adfuller
dftest = adfuller(data,autolag = 'AIC')
print('1. ADF: ', dftest[0])
print('2. P-value: ', dftest[1])
print('3. Num of Lags: ',dftest[2])
print('4. Num of Observations used for adf regression and critical values calculated: ',dftest[3])
print('5. Critical value: ')
for key, val in dftest[4].items():
    print('\t',key,':',val)

##########determine p###########
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(df_average_grade.grade.diff().dropna())
###p=1
##########determine d###########
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
# Original Series
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(df_average_grade.grade); ax1.set_title('Original Series'); ax1.axes.xaxis.set_visible(False)
ax1.set_xlim(0,500)
# 1st Differencing
ax2.plot(np.array(df_average_grade.grade.diff().values)**2); ax2.set_title('1st Order Differencing square'); ax2.axes.xaxis.set_visible(False)
ax2.set_xlim(0,500)
ax2.set_ylim(0,5)
# 2nd Differencing
ax3.plot(df_average_grade.grade.diff().diff()); ax3.set_title('2nd Order Differencing')
ax3.set_xlim(0,500)
plt.show()


from statsmodels.graphics.tsaplots import plot_acf
fig, (ax1, ax2, ax3) = plt.subplots(3,figsize=(12,8))
plot_acf(df_average_grade.grade, ax=ax1)
plot_acf(df_average_grade.grade.diff().dropna(), ax=ax2)
plot_acf(df_average_grade.grade.diff().diff().dropna(), ax=ax3)
###d=1
########determine q#########
plot_acf(df_average_grade.grade.diff().dropna())
#q=1



p,d,q = 1,1,1
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
    print(t)
    # Append test observation into overall record
    obs = test[t]
    history.append(obs)
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

fig, ax = plt.subplots(1,1,figsize=(14,8))
ax.set_xlim(0,500)
ax.plot(test,'b',label='MRA testing data')
ax.plot(predictions,'r',label='prediction from ARIMA')
ax.set_xlabel('Time index',fontsize=22)
ax.set_ylabel('Cu grade (w.t%)',fontsize=22)
# ax.set_title('ARIMA on testing MRA')
ax.legend(fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)

test_df = df_average_grade[train_size:len(df_average_grade)]



# import plotly.io as pio
# pio.renderers.default='browser'
# import plotly.graph_objects as go
# predicted = go.Scatter(
#     x=np.arange(0,1000,1),
#     y=predictions,
#     mode='lines',
#     fill=None,
#     name='Predicted Values'
#     )
# gt = go.Scatter(
#     x=np.arange(0,1000,1),
#     y=test,
#     mode='lines',
#     fill=None,
#     name='Real Values'
#     )
# data = [predicted,gt]

# fig = go.Figure(data=data)
# fig.update_layout(title='Uncertainty Quantification for MR grade test data measured every 4seconds vs Time Step',
#                     xaxis_title='Time',
#                     yaxis_title='Copper Grade (wt%)')

# fig.show()

