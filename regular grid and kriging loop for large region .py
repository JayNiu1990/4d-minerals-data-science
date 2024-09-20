import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import collections
import plotly.io as pio
from scipy.spatial import KDTree
import time
import plotly.graph_objs as go
all_regular_data = []
all_regular_neighbor_data = []
df1 = []
fields = ['BHID', 'X','Y','Z','CU']
pio.renderers.default='browser'
df_all = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core\\kmp_ddh_mra.csv", skipinitialspace=True, usecols=fields)
for ii in range(3000,3500,100):
    for jj in range(13000,13500,100):
        for kk in range(1300,1400,50):   
            df = df_all.loc[(pd.to_numeric(df_all["CU"], errors='coerce')>0.02)
                        &(pd.to_numeric(df_all["X"], errors='coerce')>=ii) & (pd.to_numeric(df_all["X"], errors='coerce')<ii+100) 
                        &(pd.to_numeric(df_all["Y"], errors='coerce')>=jj) & (pd.to_numeric(df_all["Y"], errors='coerce')<jj+100)
                        &(pd.to_numeric(df_all["Z"], errors='coerce')>=kk) & (pd.to_numeric(df_all["Z"], errors='coerce')<kk+50)]
            #df = df.dropna()
            df1.append(df)
            #df = df.reset_index(drop=True)
            print(ii,jj,kk)

            if len(df)>0:
                step=100
                all_interpolation = []
                for i in range(round(df['X'].min()),round(df['X'].max()),step):
                    for j in range(round(df['Y'].min()),round(df['Y'].max()),step):
                        for k in range(round(df['Z'].min()),round(df['Z'].max()),step):
                            df_subregion = df.loc[(pd.to_numeric(df["X"], errors='coerce')>=i) & (pd.to_numeric(df["X"], errors='coerce')<i+step) 
                                        &(pd.to_numeric(df["Y"], errors='coerce')>=j) & (pd.to_numeric(df["Y"], errors='coerce')<j+step)
                                        &(pd.to_numeric(df["Z"], errors='coerce')>=k) & (pd.to_numeric(df["Z"], errors='coerce')<k+step)]
                            df_subregion1 = df_subregion.drop(['BHID'],axis=1)
                            df_subregion1 = df_subregion1.reset_index(drop=True)
                            #print(i,j,k)
                            if len(df_subregion1)!=0:
                                
                                xx = np.linspace(i, i+step-1, step)
                                yy = np.linspace(j, j+step-1, step)
                                zz = np.linspace(k, k+step-1, step)
                                coordinate = np.vstack(np.meshgrid(xx, yy,zz, indexing='ij')).reshape(3,-1).T
                                tree = KDTree(coordinate)
                                df_subregion2 = df_subregion1.drop(['CU'],axis=1)
                                df_subregion3 = df_subregion1
                                interpolation_list = []
                                  
                                for iii in range(len(df_subregion1)):
                                    distances, points = tree.query(df_subregion2[iii:iii+1], k=1)
                                    interpolate= coordinate[points].ravel()
                                    cu = df_subregion3[iii:iii+1]['CU'].values
                                    interpolation_list.append(np.concatenate((interpolate,cu),axis=0).ravel())
                                all_interpolation.append(interpolation_list)
                                print(len(df))  
                            else:
                                pass
                flat_list = [item for sublist in all_interpolation for item in sublist]              
    
                flat_list = pd.DataFrame(flat_list)
                flat_list.columns = ['X','Y','Z','CU']
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
                
                all_regular_data.append(flat_list2)
                
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
                #OK = OrdinaryKriging3D(x1,y1,z1,cu1,variogram_model='exponential',nlags=6,exact_values=False)
                OK.display_variogram_model()
    
                gridx = np.arange(ii-1,ii+100+1,1,dtype='float64')
                gridy = np.arange(jj-1,jj+100+1,1,dtype='float64')
                gridz = np.arange(kk-1,kk+50+1,1,dtype='float64')
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
                # def returnneighbor(x3,y3,z3):
                #     ring1 = [[x3-1,y3,z3],[x3-1,y3-1,z3],[x3-1,y3+1,z3],[x3,y3-1,z3],[x3,y3+1,z3],[x3+1,y3-1,z3],[x3+1,y3,z3],[x3+1,y3+1,z3],[x3,y3,z3]]
                #     return pd.DataFrame(ring1,columns=['X','Y','Z'])
                def returnneighbor(x3,y3,z3):
                    #ring1 = [[x3-1,y3,z3],[x3-1,y3-1,z3],[x3-1,y3+1,z3],[x3,y3-1,z3],[x3,y3+1,z3],[x3+1,y3-1,z3],[x3+1,y3,z3],[x3+1,y3+1,z3],[x3,y3,z3]]
                    gridx = np.arange(x3-1,x3+2,1)
                    gridy = np.arange(y3-1,y3+2,1)
                    gridz = np.arange(z3-1,z3+2,1)
                    ring1 = np.vstack(np.meshgrid(gridx, gridy, gridz)).reshape(3,-1).T
                    return pd.DataFrame(ring1,columns=['X','Y','Z'])
    
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
                all_regular_neighbor_data.append(neighbor_nonduplicaterow)
            
            
