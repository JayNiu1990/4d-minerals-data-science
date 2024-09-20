import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import collections
import plotly.io as pio
from scipy.spatial import KDTree
import time
import plotly.graph_objs as go
##################convert to regular grid#################
fields = ['BHID', 'X','Y','Z','CU']
pio.renderers.default='browser'
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core\\kmp_ddh_mra.csv", skipinitialspace=True, usecols=fields)
df = df.loc[(pd.to_numeric(df["CU"], errors='coerce')>0.02)
            &(pd.to_numeric(df["X"], errors='coerce')>=3000) & (pd.to_numeric(df["X"], errors='coerce')<3100) 
            &(pd.to_numeric(df["Y"], errors='coerce')>=13000) & (pd.to_numeric(df["Y"], errors='coerce')<13100)
            &(pd.to_numeric(df["Z"], errors='coerce')>=1300) & (pd.to_numeric(df["Z"], errors='coerce')<1350)]

df = df.dropna()
df = df.reset_index(drop=True)
print(df['Z'].min(),df['Z'].max())
print(df['Y'].min(),df['Y'].max())
print(df['X'].min(),df['X'].max())
time1 = time.time()
# fig = go.Figure(px.scatter_3d(df, x="X",y="Y",z="Z",color='CU'))
# fig.update_traces(marker_size=6)
# fig.update_layout(font=dict(size=14))
# fig.update_layout(scene_aspectmode='data')
# fig.show()
step=100
all_interpolation = []
for k in range(int(df['Z'].min()),int(df['Z'].max()),step):
    for j in range(int(df['Y'].min()),int(df['Y'].max()),step):
        for i in range(int(df['X'].min()),int(df['X'].max()),step):
            df_subregion = df.loc[(pd.to_numeric(df["X"], errors='coerce')>=i) & (pd.to_numeric(df["X"], errors='coerce')<i+step) 
                        &(pd.to_numeric(df["Y"], errors='coerce')>=j) & (pd.to_numeric(df["Y"], errors='coerce')<j+step)
                        &(pd.to_numeric(df["Z"], errors='coerce')>=k) & (pd.to_numeric(df["Z"], errors='coerce')<k+step)]
            df_subregion1 = df_subregion.drop(['BHID'],axis=1)
            df_subregion1 = df_subregion1.reset_index(drop=True)
            #print(i,j,k)
            if len(df_subregion1)!=0:
                #print(len(df_subregion1))    
                xx = np.linspace(i, i+step-1, step)
                yy = np.linspace(j, j+step-1, step)
                zz = np.linspace(k, k+step-1, step)
                coordinate = np.vstack(np.meshgrid(xx, yy,zz, indexing='ij')).reshape(3,-1).T
                tree = KDTree(coordinate)
                df_subregion2 = df_subregion1.drop(['CU'],axis=1)
                df_subregion3 = df_subregion1
                interpolation_list = []
                for ii in range(len(df_subregion1)):
                    distances, points = tree.query(df_subregion2[ii:ii+1], k=1)
                    interpolate= coordinate[points].ravel()
                    cu = df_subregion3[ii:ii+1]['CU'].values
                    interpolation_list.append(np.concatenate((interpolate,cu),axis=0).ravel())
                all_interpolation.append(interpolation_list)
            else:
                pass
flat_list = [item for sublist in all_interpolation for item in sublist]              

flat_list = pd.DataFrame(flat_list)
flat_list.columns = ['X','Y','Z','CU']
                    #print(coordinate[points],df_subregion2[ii:ii+1])
time2 = time.time()

print(time2-time1)
flat_list.to_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core\\bore core regular subregion.csv')
flat_list1 = flat_list.groupby(['X','Y','Z'])['CU'].mean().to_frame()
flat_list1 = flat_list1.reset_index()




###########kriging##############
flat_list1['CU_log'] = np.log(flat_list1['CU'])
flat_list1['CU_log'] = round(flat_list1['CU_log'],3)


mu, sigma = 0.012, 0.011

np.random.seed(0)
noise = pd.DataFrame(np.random.normal(mu, sigma, [len(flat_list1),1])) 

noise = round(noise,4)
noise.columns = ['noise']




df_new = pd.concat([flat_list1['CU_log'],noise['noise']],axis=1)
df_new['CU_log_noise'] = df_new.sum(axis=1)
flat_list2 = pd.concat([flat_list1,df_new],axis=1)


flat_list2_random = flat_list2.sample(frac=1,random_state=10)
flat_list2_random = flat_list2_random.reset_index(drop=True)
flat_list2_random_test = flat_list2_random[int(0.8*len(flat_list2_random)):len(flat_list2_random)]

fig = go.Figure(px.scatter_3d(flat_list2_random, x="X",y="Y",z="Z",color='CU_log_noise'))
fig.update_traces(marker_size=3)
fig.update_layout(font=dict(size=14))
fig.update_layout(scene_aspectmode='data')
#fig.show()

flat_list2_random_train = flat_list2_random[0:int(0.8*len(flat_list2_random))]
x1 = np.array(flat_list2_random_train['X']).reshape(-1,1)
y1 = np.array(flat_list2_random_train['Y']).reshape(-1,1)
z1 = np.array(flat_list2_random_train['Z']).reshape(-1,1)
cu1= np.array(flat_list2_random_train['CU_log_noise'])

from sklearn.model_selection import GridSearchCV
from pykrige.rk import Krige
param_dict = {
    "method": ["ordinary3d"],
    "variogram_model": ["exponential", "gaussian", "spherical"],
    "weight": [True, False],
    "verbose": [True],
    "nlags": [10,50,100],
}

estimator = GridSearchCV(Krige(), param_dict, verbose=True)
inputdata1 = np.concatenate((x1,y1,z1),axis=1)

estimator.fit(X=inputdata1, y=cu1)

if hasattr(estimator, 'best_score_'):
    print('best_score R² = {:.3f}'.format(estimator.best_score_))
    print('best_params = ', estimator.best_params_)

print('\nCV results::')

best_para = estimator.best_params_

from pykrige.ok3d import OrdinaryKriging3D
OK = OrdinaryKriging3D(x1,y1,z1,cu1,variogram_model=best_para['variogram_model'],verbose=best_para['verbose'],weight=best_para['weight'],nlags=best_para['nlags'],exact_values=True)
#OK = OrdinaryKriging3D(x,y,z,cu,variogram_model='exponential',nlags=6,exact_values=False)
OK.display_variogram_model()

gridx = np.arange(3000-1,3100+1,1,dtype='float64')
gridy = np.arange(13000-1,13100+1,1,dtype='float64')
gridz = np.arange(1300-1,1350+1,1,dtype='float64')
zstar , ss = OK.execute('grid',gridx,gridy,gridz)

coordinate = np.vstack(np.meshgrid(gridx, gridy, gridz)).reshape(3,-1).T
coordinate = pd.DataFrame(coordinate,columns = ['X','Y','Z']).reset_index(drop=True)
cu_list = []
cu_std_list = []
for j in range(len(gridy)):
    for i in range(len(gridx)):
        for k in range(len(gridz)):
            cu_list.append(zstar[k,j,i]) ##Z Y X
            cu_std_list.append(ss[k,j,i])
            
cu_list = pd.DataFrame(cu_list,columns = ['CU_log_noise_kriging']).reset_index(drop=True)
cu_std_list = pd.DataFrame(cu_std_list,columns = ['CU_log_noise_kriging_std']).reset_index(drop=True)
kriging = pd.concat([coordinate,cu_list,cu_std_list],axis=1)


###################find neighbor coordinates#######################
from itertools import product 

def returnneighbor(x3,y3,z3):
    #ring1 = [[x3-1,y3,z3],[x3-1,y3-1,z3],[x3-1,y3+1,z3],[x3,y3-1,z3],[x3,y3+1,z3],[x3+1,y3-1,z3],[x3+1,y3,z3],[x3+1,y3+1,z3],[x3,y3,z3]]
    gridx = np.arange(x3-1,x3+2,1)
    gridy = np.arange(y3-1,y3+2,1)
    gridz = np.arange(z3-1,z3+2,1)
    ring1 = np.vstack(np.meshgrid(gridx, gridy, gridz)).reshape(3,-1).T
    return pd.DataFrame(ring1,columns=['X','Y','Z'])
# test = flat_list1[0:1].drop(['CU'],axis=1)


neighbor = []
for index, row in flat_list1.iterrows():
    x3 = row[0]
    y3 = row[1]
    z3 = row[2]
    data_srf = pd.merge(returnneighbor(x3,y3,z3), kriging, on=['X','Y','Z'],how='left')
    neighbor.append(data_srf)
neighbor = pd.concat(neighbor)
neighbor = neighbor.reset_index(drop=True)
#neighbor_duplicaterow = neighbor[neighbor.duplicated(['X','Y','Z'],keep='first')]

neighbor_nonduplicaterow = neighbor.drop_duplicates(['X','Y','Z'],keep='first')




fig = px.scatter_3d(neighbor_nonduplicaterow, x="X",y="Y",z="Z",color="CU_log_noise_kriging")
fig.update_traces(marker_size=3)
fig.update_layout(font=dict(size=18))
fig.update_layout(scene_aspectmode='data')
fig.show()





# xx1 = np.arange(3000, 3400, 1).astype('float64')
# yy1 = np.arange(13000, 13400, 1).astype('float64')
# zz1 = np.arange(1200, 1400, 1).astype('float64')

# entire_coordinate = np.vstack(np.meshgrid(xx1, yy1,zz1, indexing='ij')).reshape(3,-1).T
# entire_coordinate = pd.DataFrame(entire_coordinate,columns = ['X','Y','Z'])
# entire_coordinate['CU'] = pd.NA


# entire_coordinate = pd.merge(entire_coordinate, flat_list1, on=['X','Y','Z'],how='left')
# entire_coordinate = entire_coordinate.drop(['CU_x'],axis=1)

# entire_coordinate['CU_y'] = entire_coordinate['CU_y'].replace(np.nan, -99)

# entire_coordinate.to_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core\\bore core regular subregion 200x200x100.csv')

# ###############################test if points are in convex hull and display the results############################
# from scipy.spatial import ConvexHull, Delaunay
# from matplotlib.path import Path
# points = np.array(flat_list1.drop(['CU'],axis=1))    #bore core  X Y Z
# hull = ConvexHull(points)

# xx2 = np.arange(3000, 3400, 2).astype('float64')
# yy2 = np.arange(13000, 13400, 2).astype('float64')
# zz2 = np.arange(1200, 1400, 1).astype('float64')

# griddata = np.vstack(np.meshgrid(xx2, yy2, zz2, indexing='ij')).reshape(3,-1).T #####data for testing if a point is in convex

# def in_hull(p, hull):
#     """
#     Test if points in `p` are in `hull`

#     `p` should be a `NxK` coordinates of `N` points in `K` dimensions
#     `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
#     coordinates of `M` points in `K`dimensions for which Delaunay triangulation
#     will be computed
#     """
#     from scipy.spatial import Delaunay
#     if not isinstance(hull,Delaunay):
#         hull = Delaunay(hull)

#     return hull.find_simplex(p)>=0

# judge = in_hull(griddata,points)
# judge_df = pd.DataFrame(judge,columns = ['judge'])
# griddata_df = pd.DataFrame(griddata,columns = ['X','Y','Z'])

# vertice = pd.DataFrame(points[hull.vertices],columns = ['X','Y','Z']) ###data that forming the vertices of the convex hull

# in_convex_hull = pd.concat([griddata_df,judge_df],axis=1)
# in_convex_hull = in_convex_hull[in_convex_hull['judge']==True].drop(['judge'],axis=1) ###data in the convex hull

# in_convex_hull['label'] = 'grid data in convex hull'
# vertice['label'] = 'points that forming convex hull'

# df_combination = pd.concat([in_convex_hull,vertice])

# fig = px.scatter_3d(df_combination, x="X",y="Y",z="Z",color='label')
# fig.update_traces(marker_size=2)
# fig.update_layout(font=dict(size=18))
# fig.update_layout(scene_aspectmode='data')
# fig.show()

# ######################generate the final point cloud for spatial random forest#########################
# final_griddata = entire_coordinate.iloc[in_convex_hull.index]

# final_griddata.drop(['CU_y'],axis=1).equals(in_convex_hull.drop(['label'],axis=1))
# final_griddata = final_griddata.reset_index(drop=True)
# final_griddata.to_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core\\bore core regular coordinates in convex hull.csv')


# final_griddata1 = final_griddata.loc[(pd.to_numeric(final_griddata["CU_y"], errors='coerce')>0.02)&(pd.to_numeric(final_griddata["Z"], errors='coerce')==1300)]

# fig = px.scatter(x=final_griddata1['X'], y=final_griddata1['Y'],width=1000,height=1000)
# fig.update_xaxes(range=[3000, 3400])
# fig.update_yaxes(range=[13000, 13400])
# fig.update_traces(marker_size=20)
# fig.update_layout(font=dict(size=18))
# fig.show()



# # duplicaterow = flat_list[flat_list.duplicated(['X','Y','Z'],keep=False)]
# # row_no_duplicate = flat_list.drop(duplicaterow.index)

# # import more_itertools as mit
# # duplicaterowindex = []
# # for group in mit.consecutive_groups(duplicaterow.index):
# #     duplicaterowindex.append(list(group))
    
# # duplicaterow = duplicaterow.reset_index()

# # Mean_duplicate = []
# # for i in range(len(duplicaterowindex)):
# #     #print(duplicaterow[duplicaterow['index'].isin(duplicaterowindex[i])])
# #     #duplicaterow[duplicaterow['index'].isin(duplicaterowindex[i])][0]
# #     coordinates = duplicaterow[duplicaterow['index'].isin(duplicaterowindex[i])][0:1].drop(columns = 'CU')
# #     coordinates['CU'] = duplicaterow[duplicaterow['index'].isin(duplicaterowindex[i])]['CU'].mean()
# #     Mean_duplicate.append(coordinates)


# # Mean_duplicate = pd.concat(Mean_duplicate)

# # Mean_duplicate = Mean_duplicate.set_index('index')

# # df_final = pd.concat([row_no_duplicate,Mean_duplicate])

# # df_final = df_final.reset_index(drop=True)

# # df_final_duplicate = df_final[df_final.duplicated(['X','Y','Z'],keep=False)]
# # row_no_duplicate1 = flat_list.drop(df_final_duplicate.index)











