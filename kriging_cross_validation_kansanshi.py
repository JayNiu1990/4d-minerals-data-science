import numpy as np
import pykrige.kriging_tools as kt
import matplotlib.pyplot as plt
import pandas as pd
import collections
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import plot
import numpy as np 
fields = ['BHID', 'X','Y','Z','CU','STRAT']
pio.renderers.default='browser'
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core\\kmp_ddh_mra.csv", skipinitialspace=True, usecols=fields)
x1 = 4800
x2 = x1 + 50
y1 = 11600
y2 = y1 + 50
z1 = 1250
z2 = z1+50

df = df.dropna()
df = df.loc[(pd.to_numeric(df["CU"], errors='coerce')>0)
            &(pd.to_numeric(df["X"], errors='coerce')>x1) & (pd.to_numeric(df["X"], errors='coerce')<x2) 
            &(pd.to_numeric(df["Y"], errors='coerce')>y1) & (pd.to_numeric(df["Y"], errors='coerce')<y2)
            &(pd.to_numeric(df["Z"], errors='coerce')>z1) & (pd.to_numeric(df["Z"], errors='coerce')<z2)]
df = df.reset_index(drop=True)
df['X'] = round(df['X'],2)
df['Y'] = round(df['Y'],2)
df['Z'] = round(df['Z'],2)
mu, sigma = 0.1, 0.01

np.random.seed(0)
noise = pd.DataFrame(np.random.normal(mu, sigma, [len(df),1])) 
noise = round(noise,2)
noise.columns = ['noise']
df['CU_log'] = np.log(df['CU'])
df['CU_log'] = round(df['CU_log'],3)

df_new = pd.concat([df['CU_log'],noise['noise']],axis=1)
df_new['CU_log_noise'] = df_new.sum(axis=1)
df = pd.concat([df,df_new],axis=1)
df = df.reset_index(drop=True)


from sklearn.model_selection import LeaveOneOut
from pykrige.ok3d import OrdinaryKriging3D
loo = LeaveOneOut()
loo.get_n_splits(df)
test_gt = []
test_kriging = []

for train_index, test_index in loo.split(df):
    df_train = df.drop(index=test_index)
    df_test = df.drop(index=train_index)
    
    test_gt.append(df_test['CU_log_noise'].values[0])
    
    x_train = np.array(df_train['X']).reshape(-1,1)
    y_train = np.array(df_train['Y']).reshape(-1,1)
    z_train = np.array(df_train['Z']).reshape(-1,1)
    cu_train = np.array(df_train['CU_log_noise'])
    OK = OrdinaryKriging3D(x_train,y_train,z_train,cu_train,variogram_model='exponential',verbose=True,weight=False,nlags=50)
    
    x_test = np.array(df_test['X']).reshape(-1,1)
    y_test = np.array(df_test['Y']).reshape(-1,1)
    z_test = np.array(df_test['Z']).reshape(-1,1)
    cu_test = np.array(df_test['CU_log_noise'])
    
    zstar , ss = OK.execute('grid',x_test,y_test,z_test)
    test_kriging.append(zstar.data.ravel()[0])
    
    
plt.hist(test_gt,histtype='step',label='bore core data')
plt.hist(test_kriging,histtype='step',label='kriging')    
plt.legend()
plt.xlabel('copper grade in log')
plt.ylabel('density')   
 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
x,y = np.array(test_gt), np.array(test_kriging)
model.fit(x.reshape(-1,1), y)
r_squared = model.score(x.reshape(-1,1), y)
plt.scatter(test_gt,test_kriging)
plt.text(-1, 0.6, 'R^2 ='+ str((round(r_squared,3))))
plt.xlabel('ground truth')
plt.ylabel('kriging')

z = np.polyfit(x, y, 1)
y_hat = np.poly1d(z)(x)
plt.plot(x,y_hat,'r--')    





# fig = go.Figure(px.scatter_3d(df, x="X",y="Y",z="Z",color='CU_log_noise'))
# fig.update_traces(marker_size=3)
# fig.update_layout(scene = dict(xaxis = dict(tick0=x1,dtick=10,tickmode='linear'),
#                       yaxis = dict(tick0=x1,dtick=10,tickmode='linear'),
#                       zaxis = dict(tick0=x1,dtick=10,tickmode='linear')))
# fig.update_layout(font=dict(size=14))
# fig.update_layout(scene_aspectmode='data')
# fig.show()
























