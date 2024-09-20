###################################kriging##################################
# import numpy as np
# import pykrige.kriging_tools as kt
# from pykrige.ok3d import OrdinaryKriging3D
# from pykrige.ok import OrdinaryKriging
# import matplotlib.pyplot as plt
# import pandas as pd
# import collections
# import plotly.io as pio
# import plotly.express as px
# import plotly.graph_objs as go
# from plotly.offline import plot
# fields = ['BHID', 'X','Y','Z','CU','STRAT']
# pio.renderers.default='browser'
# df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core\\kmp_ddh_mra.csv", skipinitialspace=True, usecols=fields)

# df = df.dropna()
# df = df.loc[(pd.to_numeric(df["CU"], errors='coerce')>0.02) & 
#             (pd.to_numeric(df["X"], errors='coerce')>2500) & (pd.to_numeric(df["X"], errors='coerce')<2600) &
#             (pd.to_numeric(df["Y"], errors='coerce')>13200) & (pd.to_numeric(df["Y"], errors='coerce')<13300)
#             &(pd.to_numeric(df["Z"], errors='coerce')>1300) & (pd.to_numeric(df["Z"], errors='coerce')<1400)]
# df = df.sample(frac=1,random_state=100)
# df = df.reset_index(drop=True)

# df['CU_log'] = np.log10(df['CU'])
# df1 = df[0:1000]

# fig = go.Figure(go.Scatter3d(x=df1['X'], y=df1['Y'], z=df1['Z'],mode='markers', type='scatter3d',marker=dict(color= df1['CU_log'], showscale=True)))
# fig = go.Figure(px.scatter_3d(df, x="X",y="Y",z="Z",color=df1['CU_log']))
# fig.update_traces(marker_size=4)
# fig.update_layout(scene = dict(xaxis = dict(tick0=2200,dtick=10,tickmode='linear'),
#                      yaxis = dict(tick0=13200,dtick=10,tickmode='linear'),
#                      zaxis = dict(tick0=1300,dtick=10,tickmode='linear')))
# fig.show()


# plt.hist(df['CU'],bins=np.arange(0,5,0.1))
# # fig = px.scatter_3d(df, x="X",y="Y",z="Z",color="CU")
# # fig.update_traces(marker_size=3)
# # fig.show()
# x = np.array(round(df['X'],1))
# y = np.array(round(df['Y'],1))
# cu = np.array(round(df['CU_log'],3))
# OK = OrdinaryKriging(x,y,cu,variogram_model='gaussian',verbose=True,enable_plotting=True,nlags=10)

# gridx = np.arange(x.min(),x.max(),0.2,dtype='float64')
# gridy = np.arange(y.min(),y.max(),0.2,dtype='float64')
# zstar , ss = OK.execute('grid',gridx,gridy)

# #matrix[gridz,gridy,gridx] = zstar.data.reshape(-1)
# coordinate = np.vstack(np.meshgrid(gridx, gridy)).reshape(2,-1).T
# coordinate = pd.DataFrame(coordinate,columns = ['X','Y']).reset_index(drop=True)
# cu_list = []
# for j in range(len(gridy)):
#     for i in range(len(gridx)):
#             cu_list.append(zstar[j,i]) ##Z Y X
            
# cu_list = pd.DataFrame(cu_list,columns = ['CU_log_kriging']).reset_index(drop=True)
# kriging = pd.concat([coordinate,cu_list],axis=1)
# kriging1 = kriging.loc[(pd.to_numeric(kriging["X"], errors='coerce')>2537.6) & (pd.to_numeric(kriging["X"], errors='coerce')<2537.9)
#                        &(pd.to_numeric(kriging["Y"], errors='coerce')>13066) & (pd.to_numeric(kriging["Y"], errors='coerce')<13066.3)]




# cax = plt.scatter(df1['X'],df1['Y'],c=df1['CU_log'])
# cbar = plt.colorbar(cax, fraction=0.03)

# cax = plt.imshow(zstar, extent=(2500, 2600, 13000, 13100), origin='lower')
# plt.scatter(x, y, c='k', marker='.')
# cbar=plt.colorbar(cax)
# plt.title('Porosity estimate')


# plt.hist(zstar.data.reshape(-1),bins=np.arange(-2,1,0.1),density=True)

# plt.hist(df1['CU_log'],bins=np.arange(-2,1,0.1),density=True)


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(df1['X'],df1['Y'],df1['Z'])


# kriging1 = kriging.loc[(pd.to_numeric(kriging["X"], errors='coerce')>2828) & (pd.to_numeric(kriging["X"], errors='coerce')<2830)
#                        &(pd.to_numeric(kriging["Y"], errors='coerce')>13575) & (pd.to_numeric(kriging["Y"], errors='coerce')<13577)
#                        &(pd.to_numeric(kriging["Z"], errors='coerce')>1398) & (pd.to_numeric(kriging["Z"], errors='coerce')<1400)]




###########################################kriging 3D##################################################
import numpy as np
import pykrige.kriging_tools as kt
from pykrige.ok3d import OrdinaryKriging3D
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import pandas as pd
import collections
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import plot
import numpy as np 
fields = ['BHID', 'X','Y','Z','CuT_dh']
pio.renderers.default='browser'
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Escondida Trucks\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields,dtype='unicode')

x1 = 17100
x2 = x1 + 100
y1 = 106300
y2 = y1 + 100
z1 = 2700
z2 = z1+100

# x1=3700
# x2=3720
# y1=13100
# y2=13120
# z1=1300
# z2=1400
df = df.loc[(pd.to_numeric(df["CuT_dh"], errors='coerce')>0.02)]
df['X'] = pd.to_numeric(df['X'])
df['Y'] = pd.to_numeric(df['Y'])
df['Z'] = pd.to_numeric(df['Z'])
df['CuT_dh'] = pd.to_numeric(df['CuT_dh'])


df['X'] = round(df['X'],2)
df['Y'] = round(df['Y'],2)
df['Z'] = round(df['Z'],2)
# df = df.dropna()

df = df.loc[(pd.to_numeric(df["CuT_dh"], errors='coerce')>0.02) &
            (pd.to_numeric(df["X"], errors='coerce')>x1) & (pd.to_numeric(df["X"], errors='coerce')<x2) 
            &(pd.to_numeric(df["Y"], errors='coerce')>y1) & (pd.to_numeric(df["Y"], errors='coerce')<y2)
            &(pd.to_numeric(df["Z"], errors='coerce')>z1) & (pd.to_numeric(df["Z"], errors='coerce')<z2)]

fig = go.Figure(px.scatter_3d(df, x="X",y="Y",z="Z",color='CuT_dh'))
fig.update_traces(marker_size=4)
#fig.update_layout(font=dict(size=14))
fig.show()


df = df.loc[(pd.to_numeric(df["CuT_dh"], errors='coerce')>0.02)]
df = df.reset_index(drop=True)
# add gaussian noise
df['X'] = round(df['X'],2)
df['Y'] = round(df['Y'],2)
df['Z'] = round(df['Z'],2)
mu, sigma = 0.1, 0.01

np.random.seed(1000)
noise = pd.DataFrame(np.random.normal(mu, sigma, [len(df),1])) 

noise = round(noise,2)
noise.columns = ['noise']

df_new = pd.concat([df['CuT_dh'],noise['noise']],axis=1)
df_new['CuT_dh_noise'] = df_new.sum(axis=1)
df = pd.concat([df,df_new],axis=1)

df['CuT_dh_log_noise'] = np.log10(df['CuT_dh_noise'])
df['CuT_dh_log_noise'] = round(df['CuT_dh_log_noise'],3)

df = df.sample(frac=1,random_state=100)
df = df.reset_index(drop=True)
df_test = df[int(0.8*len(df)):len(df)]
#fig = go.Figure(go.Scatter3d(x=df['X'], y=df['Y'], z=df['Z'],mode='markers', type='scatter3d',marker=dict(color= df['CU_log'], showscale=True)))
fig = go.Figure(px.scatter_3d(df, x="X",y="Y",z="Z",color='CuT_dh_log_noise'))
fig.update_traces(marker_size=4)
fig.update_layout(scene = dict(xaxis = dict(tick0=x1,dtick=10,tickmode='linear'),
                     yaxis = dict(tick0=x1,dtick=10,tickmode='linear'),
                     zaxis = dict(tick0=x1,dtick=10,tickmode='linear')))
#fig.update_layout(font=dict(size=14))
fig.show()





# df = df.groupby('BHID').filter(lambda x:len(x)>30)
# df = df.loc[(pd.to_numeric(df["CU"], errors='coerce')>0.02)]

df = df[0:int(0.8*len(df))]
x = np.array(df['X'])
y = np.array(df['Y'])
z = np.array(df['Z'])
cu = np.array(df['CuT_dh_log_noise'])
OK = OrdinaryKriging3D(x,y,z,cu,variogram_model='spherical',verbose=True,enable_plotting=True,nlags=20)

#df1 = df[0:200]
gridx = np.arange(x1,x2,1,dtype='float64')
gridy = np.arange(y1,y2,1,dtype='float64')
gridz = np.arange(z1,z2,1,dtype='float64')
zstar , ss = OK.execute('grid',gridx,gridy,gridz)

coordinate = np.vstack(np.meshgrid(gridx, gridy, gridz)).reshape(3,-1).T
coordinate = pd.DataFrame(coordinate,columns = ['X','Y','Z']).reset_index(drop=True)
cu_list = []
for j in range(len(gridy)):
    for i in range(len(gridx)):
        for k in range(len(gridz)):
            cu_list.append(zstar[k,j,i]) ##Z Y X
            
cu_list = pd.DataFrame(cu_list,columns = ['CU_log_noise_kriging']).reset_index(drop=True)
kriging = pd.concat([coordinate,cu_list],axis=1)

# kriging= kriging.loc[(pd.to_numeric(kriging["X"], errors='coerce')>x1) & (pd.to_numeric(kriging["X"], errors='coerce')<x2) &
#             (pd.to_numeric(kriging["Y"], errors='coerce')>y1) & (pd.to_numeric(kriging["Y"], errors='coerce')<y2)
#             &(pd.to_numeric(kriging["Z"], errors='coerce')>z1) & (pd.to_numeric(kriging["Z"], errors='coerce')<z2)]
fig = go.Figure(px.scatter_3d(kriging[0:10000], x="X",y="Y",z="Z",color='CU_log_noise_kriging'))
fig.update_traces(marker_size=4)
fig.update_layout(scene = dict(xaxis = dict(tick0=2500,dtick=10,tickmode='linear'),
                      yaxis = dict(tick0=13200,dtick=10,tickmode='linear'),
                      zaxis = dict(tick0=1300,dtick=10,tickmode='linear'))) 
fig.show()

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
plt.text(-0.8, 0.8, 'R^2 ='+ str((round(r_squared,3))))
plt.xlabel('ground truth')
plt.ylabel('kriging')

z = np.polyfit(x, y, 1)
y_hat = np.poly1d(z)(x)
plt.plot(x,y_hat,'r--')


plt.hist(df['CU_log_noise'],density=True,bins=20,histtype='step',label='bore core data')
plt.hist(kriging['CU_log_noise_kriging'],bins=50,density=True,histtype='step',label='kriging inference')      
plt.xlabel('Copper grade (log)')    
plt.ylabel('Density')
plt.legend()
# plt.xlim(-1.3,1)
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
plt.hist(truck_mean,bins=np.arange(0,2,0.05),density=True,color='b')
plt.xlabel('Truck copper grade log')
plt.ylabel('Density')
plt.show()
partial_truck = []
partial_truck_mean = []

for i in range(len(truck)):
    np.random.shuffle(truck[i])
    partial_truck.append(truck[i][0:13])
    partial_truck_mean.append(truck[i][0:13].mean())

plt.hist(partial_truck_mean,bins=np.arange(0,2,0.05),density=True,color='b')
plt.xlabel('20% Truck copper grade log')
plt.ylabel('Density')
plt.show()
#np.random.shuffle(sample)

plt.plot(sample,color='b')
plt.xlabel('time step')
plt.ylabel('simulated copper grade on belt')
plt.xlim(100,200)

kriging1 = kriging.loc[(pd.to_numeric(kriging["Z"], errors='coerce')<1330)]
fig = go.Figure(px.scatter_3d(kriging1, x="X",y="Y",z="Z",color='CU_log_noise_kriging'))
fig.update_traces(marker_size=4)
fig.show()
plt.hist(df['CU_log_noise'],density=True,bins=50,histtype='step',label='bore core data')
plt.hist(kriging1['CU_log_noise_kriging'],bins=50,density=True,histtype='step',label='kriging inference')    





