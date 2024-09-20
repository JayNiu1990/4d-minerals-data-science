###########################################kriging 3D##################################################
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd    
import plotly.io as pio
import plotly.graph_objs as go
from scipy import stats
import random
fields = ['X','Y','Z','CU'] 
pio.renderers.default='browser'
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\bore core regular.csv", skipinitialspace=True, usecols=fields,dtype='unicode')
x1 = 3000
x2 = x1 + 100
y1 = 12500
y2 = y1 + 100
z1 = 1300
z2 = z1+100
df = df.dropna()
df = df.loc[(pd.to_numeric(df["CU"], errors='coerce')>0)]
df['X'] = pd.to_numeric(df['X'])
df['Y'] = pd.to_numeric(df['Y'])
df['Z'] = pd.to_numeric(df['Z'])
df['CU'] = pd.to_numeric(df['CU'])


df['X'] = round(df['X'],2)
df['Y'] = round(df['Y'],2)
df['Z'] = round(df['Z'],2)


df = df.loc[(pd.to_numeric(df["X"], errors='coerce')>=x1) & (pd.to_numeric(df["X"], errors='coerce')<x2) 
            &(pd.to_numeric(df["Y"], errors='coerce')>=y1) & (pd.to_numeric(df["Y"], errors='coerce')<y2)
            &(pd.to_numeric(df["Z"], errors='coerce')>=z1) & (pd.to_numeric(df["Z"], errors='coerce')<z2)]

df = df.reset_index(drop=True)
# mu, sigma = 0.1, 0.01

# np.random.seed(0)
# noise = pd.DataFrame(np.random.normal(mu, sigma, [len(df),1])) 

# noise = round(noise,2)
# noise.columns = ['noise']
# df['CU_log'] = np.log(df['CU'])
# df['CU_log'] = round(df['CU_log'],3)


# df_new = pd.concat([df['CU_log'],noise['noise']],axis=1)
# df_new['CU_log_noise'] = df_new.sum(axis=1)
# df = pd.concat([df,df_new],axis=1)


# df = df.sample(frac=1,random_state=10)
# df = df.reset_index(drop=True)
#df_test = df[int(0.8*len(df)):len(df)]

fig = px.scatter_3d(df, x="X",y="Y",z="Z",color="CU")
fig.update_traces(marker_size=4)
fig.update_layout(font=dict(size=16))
fig.update_layout(scene_aspectmode='data')
fig.show()  
#df = df.loc[pd.to_numeric(pd.to_numeric(df["Z"], errors='coerce')==1300)]



# df = df.groupby('BHID').filter(lambda x:len(x)>30)
# df = df.loc[(pd.to_numeric(df["CU"], errors='coerce')>0.02)]
# fig = px.scatter_3d(df, x="X",y="Y",z="Z",color="CU")
# fig.update_traces(marker_size=2)
# fig.show()
#df = df[0:int(0.8*len(df))]
x = np.array(df['X']).reshape(-1,1)
y = np.array(df['Y']).reshape(-1,1)
z = np.array(df['Z']).reshape(-1,1)
cu = np.array(df['CU'])

from sklearn.model_selection import GridSearchCV
from pykrige.rk import Krige
# param_dict = {
#     "method": ["ordinary3d"],
#     "variogram_model": ["exponential", "gaussian", "spherical"],
#     "weight": [True, False],
#     "verbose": [True],
#     "nlags": [10,50,100],
# }

# estimator = GridSearchCV(Krige(), param_dict, verbose=True)
# inputdata = np.concatenate((x,y,z),axis=1)

# estimator.fit(X=inputdata, y=cu)

# if hasattr(estimator, 'best_score_'):
#     print('best_score R² = {:.3f}'.format(estimator.best_score_))
#     print('best_params = ', estimator.best_params_)

# print('\nCV results::')

# best_para = estimator.best_params_

from pykrige.ok3d import OrdinaryKriging3D
#OK = OrdinaryKriging3D(x,y,z,cu,variogram_model=best_para['variogram_model'],verbose=best_para['verbose'],weight=best_para['weight'],nlags=best_para['nlags'],exact_values=True)
OK = OrdinaryKriging3D(x,y,z,cu,variogram_model='gaussian',nlags=6,exact_values=False)
OK.display_variogram_model()
#df1 = df[0:200]
gridx = np.arange(x1,x2,2,dtype='float64')
gridy = np.arange(y1,y2,2,dtype='float64')
gridz = np.arange(z1,z2,2,dtype='float64')
zstar , ss = OK.execute('grid',gridx,gridy,gridz,backend="loop",n_closest_points=10)

coordinate = np.vstack(np.meshgrid(gridx, gridy, gridz)).reshape(3,-1).T
coordinate = pd.DataFrame(coordinate,columns = ['X','Y','Z']).reset_index(drop=True)
cu_list = []
cu_std_list = []
for j in range(len(gridy)):
    for i in range(len(gridx)):
        for k in range(len(gridz)):
            cu_list.append(zstar[k,j,i]) ##Z Y X
            cu_std_list.append(ss[k,j,i])
            
cu_list = pd.DataFrame(cu_list,columns = ['CU_kriging']).reset_index(drop=True)
cu_std_list = pd.DataFrame(cu_std_list,columns = ['CU_kriging_std']).reset_index(drop=True)
kriging = pd.concat([coordinate,cu_list,cu_std_list],axis=1)

# df1 = df.drop(['CU','CU_log','CU_log','noise'],axis=1)
# df1.loc[len(df1)]=[3090,13090,1305,-0.5]

fig = px.scatter_3d(kriging, x="X",y="Y",z="Z",color='CU_kriging')
fig.update_traces(marker_size=2)
# fig.update_layout(scene = dict(xaxis = dict(tick0=3500,dtick=10,tickmode='linear'),
#                       yaxis = dict(tick0=12500,dtick=10,tickmode='linear'),
#                       zaxis = dict(tick0=1295,dtick=10,tickmode='linear'))) 
fig.update_layout(scene_aspectmode='data')
fig.show()










###################add one point####################
# kriging= kriging.loc[(pd.to_numeric(kriging["X"], errors='coerce')>x1) & (pd.to_numeric(kriging["X"], errors='coerce')<x2) &
#             (pd.to_numeric(kriging["Y"], errors='coerce')>y1) & (pd.to_numeric(kriging["Y"], errors='coerce')<y2)
#             &(pd.to_numeric(kriging["Z"], errors='coerce')>z1) & (pd.to_numeric(kriging["Z"], errors='coerce')<z2)]





xx1 = np.arange(x1, x2, 50).astype('float64')
yy1 = np.arange(y1, y2, 50).astype('float64')
zz1 = np.arange(z1, z2, 10).astype('float64')

blocks = []
for k in zz1:
    for j in yy1:
        for i in xx1:
            sub_block = kriging.loc[(pd.to_numeric(kriging["X"], errors='coerce')>=i) & (pd.to_numeric(kriging["X"], errors='coerce')<i+20) &
                         (pd.to_numeric(kriging["Y"], errors='coerce')>=j) & (pd.to_numeric(kriging["Y"], errors='coerce')<j+20)
                         &(pd.to_numeric(kriging["Z"], errors='coerce')>=k) & (pd.to_numeric(kriging["Z"], errors='coerce')<k+10)]
            blocks.append(sub_block)

blocks_borecore = []
for k in zz1:
    for j in yy1:
        for i in xx1:
            sub_block = df.loc[(pd.to_numeric(df["X"], errors='coerce')>=i) & (pd.to_numeric(df["X"], errors='coerce')<i+20) &
                         (pd.to_numeric(df["Y"], errors='coerce')>=j) & (pd.to_numeric(df["Y"], errors='coerce')<j+20)
                         &(pd.to_numeric(df["Z"], errors='coerce')>=k) & (pd.to_numeric(df["Z"], errors='coerce')<k+10)]
            blocks_borecore.append(sub_block)

fig = go.Figure(px.scatter_3d(blocks[0], x="X",y="Y",z="Z"))
fig.update_traces(marker_size=6)
fig.update_layout(scene = dict(xaxis = dict(tick0=2500,dtick=10,tickmode='linear'),
                      yaxis = dict(tick0=13200,dtick=10,tickmode='linear'),
                      zaxis = dict(tick0=1300,dtick=10,tickmode='linear'))) 
fig.update_layout(scene_aspectmode='data')
fig.show()

xx2 = np.arange(3010, 3100, 50).astype('float64')
yy2 = np.arange(13010, 13100, 50).astype('float64')
zz2 = np.arange(1305, 1350, 10).astype('float64')
centre_simulated_bore_core_coordinate = np.vstack(np.meshgrid(xx2, yy2, zz2, indexing='ij')).reshape(3,-1).T
centre_simulated_bore_core_coordinate = pd.DataFrame(centre_simulated_bore_core_coordinate,columns=['X','Y','Z'])

centre_simulated_bore_core_coordinate = pd.merge(centre_simulated_bore_core_coordinate, kriging, on=['X','Y','Z'],how='left')

coordinates = []
for i in range(len(blocks)):
    list2 = []
    list2.append(int(blocks[i]['X'].mean()+0.5))
    list2.append(int(blocks[i]['Y'].mean()+0.5))
    list2.append(int(blocks[i]['Z'].mean()+0.5))
    list2.append(blocks[i]['CU_log_noise_kriging'].mean())
    coordinates.append(list2)
coordinates = pd.DataFrame(coordinates,columns=['X','Y','Z','CU_log_noise_kriging'])
coordinates_update = coordinates
coordinates_update.loc[24] = [3090,13090,1305,-0.5]

fig = go.Figure(px.scatter_3d(coordinates_update, x="X",y="Y",z="Z",color='CU_log_noise_kriging'))
fig.update_traces(marker_size=6)
fig.update_layout(scene = dict(xaxis = dict(tick0=2500,dtick=10,tickmode='linear'),
                      yaxis = dict(tick0=13200,dtick=10,tickmode='linear'),
                      zaxis = dict(tick0=1300,dtick=10,tickmode='linear'))) 
fig.update_layout(scene_aspectmode='data')
fig.show()



from pykrige.ok3d import OrdinaryKriging3D
x1 = np.array(coordinates['X']).reshape(-1,1)
y1 = np.array(coordinates['Y']).reshape(-1,1)
z1 = np.array(coordinates['Z']).reshape(-1,1)
cu1 = np.array(coordinates['CU_log_noise_kriging'])
OK1 = OrdinaryKriging3D(x1,y1,z1,cu1,variogram_model=best_para['variogram_model'],verbose=best_para['verbose'],weight=best_para['weight'],nlags=best_para['nlags'],exact_values=True)
#OK = OrdinaryKriging3D(x,y,z,cu,variogram_model='exponential',nlags=6,exact_values=False)
OK1.display_variogram_model()



gridx = np.arange(x1,x2,5,dtype='float64')
gridy = np.arange(y1,y2,5,dtype='float64')
gridz = np.arange(z1,z2,5,dtype='float64')
zstar1 , ss1 = OK1.execute('grid',gridx,gridy,gridz)

coordinate1 = np.vstack(np.meshgrid(gridx, gridy, gridz)).reshape(3,-1).T
coordinate1 = pd.DataFrame(coordinate1,columns = ['X','Y','Z']).reset_index(drop=True)
cu_list1 = []
cu_std_list1 = []
for j in range(len(gridy)):
    for i in range(len(gridx)):
        for k in range(len(gridz)):
            cu_list1.append(zstar1[k,j,i]) ##Z Y X
            cu_std_list1.append(ss1[k,j,i])
            
cu_list1 = pd.DataFrame(cu_list1,columns = ['CU_log_noise_kriging']).reset_index(drop=True)
cu_std_list1 = pd.DataFrame(cu_std_list1,columns = ['CU_log_noise_kriging_std']).reset_index(drop=True)
kriging1 = pd.concat([coordinate1,cu_list1,cu_std_list1],axis=1)

fig = go.Figure(px.scatter_3d(kriging1, x="X",y="Y",z="Z",color='CU_log_noise_kriging'))
fig.update_traces(marker_size=6)
fig.update_layout(scene = dict(xaxis = dict(tick0=2500,dtick=10,tickmode='linear'),
                      yaxis = dict(tick0=13200,dtick=10,tickmode='linear'),
                      zaxis = dict(tick0=1300,dtick=10,tickmode='linear'))) 
fig.update_layout(scene_aspectmode='data')
fig.show()




from pykrige.ok3d import OrdinaryKriging3D
x2 = np.array(coordinates_update['X']).reshape(-1,1)
y2 = np.array(coordinates_update['Y']).reshape(-1,1)
z2 = np.array(coordinates_update['Z']).reshape(-1,1)
cu2 = np.array(coordinates_update['CU_log_noise_kriging'])
OK2 = OrdinaryKriging3D(x2,y2,z2,cu2,variogram_model=best_para['variogram_model'],verbose=best_para['verbose'],weight=best_para['weight'],nlags=best_para['nlags'],exact_values=True)
#OK = OrdinaryKriging3D(x,y,z,cu,variogram_model='exponential',nlags=6,exact_values=False)
OK2.display_variogram_model()

gridx = np.arange(x1,x2,50,dtype='float64')
gridy = np.arange(y1,y2,50,dtype='float64')
gridz = np.arange(z1,z2,10,dtype='float64')
zstar2 , ss2 = OK2.execute('grid',gridx,gridy,gridz)

coordinate2 = np.vstack(np.meshgrid(gridx, gridy, gridz)).reshape(3,-1).T
coordinate2 = pd.DataFrame(coordinate2,columns = ['X','Y','Z']).reset_index(drop=True)
cu_list2 = []
cu_std_list2 = []
for j in range(len(gridy)):
    for i in range(len(gridx)):
        for k in range(len(gridz)):
            cu_list2.append(zstar2[k,j,i]) ##Z Y X
            cu_std_list2.append(ss2[k,j,i])
            
cu_list2 = pd.DataFrame(cu_list2,columns = ['CU_log_noise_kriging']).reset_index(drop=True)
cu_std_list2 = pd.DataFrame(cu_std_list2,columns = ['CU_log_noise_kriging_std']).reset_index(drop=True)
kriging2 = pd.concat([coordinate2,cu_list2,cu_std_list2],axis=1)

fig = go.Figure(px.scatter_3d(kriging2, x="X",y="Y",z="Z",color='CU_log_noise_kriging'))
fig.update_traces(marker_size=6)
fig.update_layout(scene = dict(xaxis = dict(tick0=x1,dtick=10,tickmode='linear'),
                      yaxis = dict(tick0=y1,dtick=10,tickmode='linear'),
                      zaxis = dict(tick0=z1,dtick=10,tickmode='linear'))) 
fig.update_layout(scene_aspectmode='data')
fig.show()
















'''
kriging1 = kriging[kriging['Z']==1250]
kriging2 = kriging1.pivot('Y', 'X', 'CU_log_noise_kriging').values
kriging2_std = kriging1.pivot('Y', 'X', 'CU_log_noise_kriging_std').values
# plt.imshow(kriging2, cmap='hot', interpolation='nearest')
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(121)
im1 = ax1.imshow(kriging2, cmap='hot', interpolation='nearest')
fig.colorbar(im1)
ax2 = fig.add_subplot(122)
im2 = ax2.imshow(kriging2_std, cmap='hot', interpolation='nearest')
fig.colorbar(im2)



plt.show()


kriging1 = kriging[kriging['Z']==1298]
kriging2 = kriging1.pivot('Y', 'X', 'CU_log_noise_kriging').values
plt.imshow(kriging2, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()
# kriging1 = kriging.loc[(pd.to_numeric(kriging["X"], errors='coerce')>2513) & (pd.to_numeric(kriging["X"], errors='coerce')<2515)
#                        &(pd.to_numeric(kriging["Y"], errors='coerce')>13067) & (pd.to_numeric(kriging["Y"], errors='coerce')<13069)
#                        &(pd.to_numeric(kriging["Z"], errors='coerce')>1387) & (pd.to_numeric(kriging["Z"], errors='coerce')<1388)]
df2_test = df_test.drop(['BHID','CU','STRAT'],axis=1)
# a = np.arange(2500,2600,100,dtype='float64')
# b = np.arange(13200,13300,100,dtype='float64')
# c = np.arange(1300,1400,100,dtype='float64')
df2_test['CU_log_noise_kriging'] = ' '
CU_log_kriging = []
for i in range(len(df2_test)):
    a = df2_test[i:i+1]['X'].to_frame().values.ravel()
    b = df2_test[i:i+1]['Y'].to_frame().values.ravel()
    c = df2_test[i:i+1]['Z'].to_frame().values.ravel()
    zstar1 , ss1 = OK.execute('grid',a,b,c)
    CU_log_kriging.append(zstar1.data.ravel()[0])

    #print(zstar1.data.ravel()[0])
df2_test['CU_log_noise_kriging'] = CU_log_kriging






from sklearn.linear_model import LinearRegression
model = LinearRegression()
x,y = np.array(df2_test['CU_log_noise']), np.array(df2_test['CU_log_noise_kriging'])
model.fit(x.reshape(-1,1), y)
r_squared = model.score(x.reshape(-1,1), y)
plt.scatter(df2_test['CU_log_noise'],df2_test['CU_log_noise_kriging'])
plt.text(-1, 0.6, 'R^2 ='+ str((round(r_squared,3))))
plt.xlabel('ground truth')
plt.ylabel('kriging')

z = np.polyfit(x, y, 1)
y_hat = np.poly1d(z)(x)
plt.plot(x,y_hat,'r--')


plt.hist(df['CU_log_noise'],density=True,bins=25,histtype='step',label='bore core data')
plt.hist(kriging['CU_log_noise_kriging'],bins=25,density=True,histtype='step',label='kriging inference')      
plt.xlabel('Copper grade (log)')    
plt.ylabel('Density')
plt.legend()
# plt.legend()

blocks= kriging.loc[(pd.to_numeric(kriging["Y"], errors='coerce')>=11820) & (pd.to_numeric(kriging["Y"], errors='coerce')<=11829)
                    &(pd.to_numeric(kriging["Z"], errors='coerce')>=1320) & (pd.to_numeric(kriging["Z"], errors='coerce')<=1349)]
blocks['CU'] = 10**blocks['CU_log_noise_kriging']

from sklearn.mixture import GaussianMixture
plt.hist(blocks['CU'],bins=np.arange(0,2,0.02),density=True,color='blue')
plt.xlabel('Copper grade log')
plt.ylabel('Density')

blocks1 = np.array(blocks['CU'])
blocks1 = blocks1.reshape((blocks1.shape[0], 1))

K = 8
from scipy.stats import norm
def mix_pdf(x, loc, scale, weights):
    d = np.zeros_like(x)
    print(d.shape)
    for mu, sigma, pi in zip(loc, scale, weights):
        d += pi * norm.pdf(x, loc=mu, scale=sigma)
    return d

mix = GaussianMixture(n_components=K, random_state=1, max_iter=100).fit(blocks1)
pi, mu, sigma = mix.weights_.flatten(), mix.means_.flatten(), np.sqrt(mix.covariances_.flatten())

grid = np.arange(np.min(blocks1), np.max(blocks1), 0.01)

plt.hist(blocks1, bins=np.arange(0,2,0.02), density=True, color='b')
plt.plot(grid, mix_pdf(grid, mu, sigma, pi),color='r')
plt.xlabel('Block model copper grade log')
plt.ylabel('Density')

sample = mix.sample(9000)[0]
truck = []
for i in range(0,9000,67):
    truck.append(sample[i:i+67])
    
truck_mean = []
for i in range(len(truck)):
    truck_mean.append(truck[i].mean())
plt.hist(truck_mean,bins=np.arange(0,2,0.2),density=True,color='b')
plt.xlabel('Truck copper grade log')
plt.ylabel('Density')
plt.show()
partial_truck = []
partial_truck_mean = []

for i in range(len(truck)):
    np.random.shuffle(truck[i])
    partial_truck.append(truck[i][0:13])
    partial_truck_mean.append(truck[i][0:13].mean())

plt.hist(partial_truck_mean,bins=np.arange(0,2,0.2),density=True,color='b')
plt.xlabel('20% Truck copper grade log')
plt.ylabel('Density')
plt.show()
#np.random.shuffle(sample)

plt.plot(sample,color='b')
plt.xlabel('time step')
plt.ylabel('simulated copper grade on belt')
plt.xlim(0,200)

kriging1 = kriging.loc[(pd.to_numeric(kriging["Z"], errors='coerce')<1330)]
fig = go.Figure(px.scatter_3d(kriging1, x="X",y="Y",z="Z",color='CU_log_noise_kriging'))
fig.update_traces(marker_size=4)
fig.show()
plt.hist(df['CU_log_noise'],density=True,bins=50,histtype='step',label='bore core data')
plt.hist(kriging1['CU_log_noise_kriging'],bins=50,density=True,histtype='step',label='kriging inference')    

###############directional variogram###############
import geostatspy.GSLIB as GSLIB                          # GSLIB utilities, viz and wrapped functions
import geostatspy.geostats as geostats                    # GSLIB converted to Python
import matplotlib.pyplot as plt   

import scipy.stats                                        # summary stats of ndarrays
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import plot

fields = ['X','Y','Z','CU']
pio.renderers.default='browser'
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core\\bore core regular.csv", skipinitialspace=True, usecols=fields)
x1 = 0
x2 = x1 + 20000
y1 = 0
y2 = y1 + 20000
z1 = 1300
z2 = z1+1
df = df.dropna()
df = df.loc[(pd.to_numeric(df["CU"], errors='coerce')>0.02)
            &(pd.to_numeric(df["X"], errors='coerce')>=x1) & (pd.to_numeric(df["X"], errors='coerce')<x2) 
            &(pd.to_numeric(df["Y"], errors='coerce')>=y1) & (pd.to_numeric(df["Y"], errors='coerce')<y2)
            &(pd.to_numeric(df["Z"], errors='coerce')>=z1) & (pd.to_numeric(df["Z"], errors='coerce')<z2)]
df = df.reset_index(drop=True)

tmin = -9999.; tmax = 9999.                             # no trimming 
lag_dist = 10.0; lag_tol = 5.0; nlag = 20;            # maximum lag is 700m and tolerance > 1/2 lag distance for smoothing
bandh = 9999.9; atol = 20                             # no bandwidth, directional variograms
isill = 1                                               # standardize sill
azi_mat = [0,90]           # directions in azimuth to consider
# Arrays to store the results
lag = np.zeros((len(azi_mat),nlag+2)); gamma = np.zeros((len(azi_mat),nlag+2)); npp = np.zeros((len(azi_mat),nlag+2));
for iazi in range(0,len(azi_mat)):                      # Loop over all directions
    lag[iazi,:], gamma[iazi,:], npp[iazi,:] = geostats.gamv(df,"X","Y","CU",tmin,tmax,lag_dist,lag_tol,nlag,azi_mat[iazi],atol,bandh,isill)
    plt.subplot(4,2,iazi+1)
    plt.plot(lag[iazi,:],gamma[iazi,:],'x',color = 'black',label = 'Azimuth ' +str(azi_mat[iazi]))
    plt.xlabel(r'Lag Distance $\bf(h)$, (m)')
    plt.ylabel(r'$\gamma \bf(h)$')
    plt.title('Directional NSCORE Porosity Variogram')
    plt.xlim([0,200])
    plt.ylim([0,1.8])
    plt.legend(loc='upper left')
    plt.grid(True)
plt.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=4.2, wspace=0.2, hspace=0.3)
plt.show()
'''

