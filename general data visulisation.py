if __name__ == '__main__':
    import warnings
    import matplotlib.pyplot as plt
    import plotly.express as px
    import numpy as np
    import pandas as pd    
    import plotly.io as pio
    import plotly.graph_objs as go
    import seaborn as sns
    pio.renderers.default='browser'
    fields = ['MIDX|','MIDY|','MIDZ|','NI|']
    df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Murrin Murrin\\m08p06gc_drilling.csv", skipinitialspace=True, usecols=fields, low_memory=False)
    df = df[(pd.to_numeric(df["NI|"], errors='coerce')>0)]
    df['NI_LOG|'] = np.log(df['NI|'])
    df.head()
    
    fig = px.scatter_3d(df, x="MIDX|",y="MIDY|",z="MIDZ|",color="NI_LOG|")
    fig.update_traces(marker_size=2)
    #fig.update_layout(font=dict(size=16))
    fig.update_layout(scene_aspectmode='data')  
    fig.update_scenes(xaxis = dict(title = 'X', tickfont = dict(size = 12), titlefont = dict(size = 18)))
    fig.update_scenes(yaxis = dict(title = 'Y', tickfont = dict(size = 12), titlefont = dict(size = 18)))
    fig.update_scenes(zaxis = dict(title = 'Z', tickfont = dict(size = 12), titlefont = dict(size = 18)))
    fig.show()

    #sub_df = df[(pd.to_numeric(df["MIDX|"], errors='coerce')==7300.048) & (pd.to_numeric(df["MIDY|"], errors='coerce')==19100.076)& (pd.to_numeric(df["MIDZ|"], errors='coerce')==456.76)]



    plt.hist(np.log(df['ni']),bins=np.arange(-4,4,0.1),color='b')
    plt.xlabel('log Ni',fontsize=14)
    plt.ylabel('frequency',fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.title('histogram of log Ni in Murrin Murrin',fontsize=14)
    
    df2 = df.loc[df['ni']!=0.027]
    plt.hist(np.log(df2['ni']),bins=np.arange(-4,4,0.1),color='b')
    plt.xlabel('log Ni',fontsize=14)
    plt.ylabel('frequency',fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.title('histogram of log Ni in Murrin Murrin without Ni=0.027',fontsize=14)
    
    #g =  sns.pairplot(df, diag_kind = 'kde')
    df1 = df[['ni_log','co_log','mg_log','al_log','fe_log','si_log']]
    g = sns.PairGrid(df1)
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot) # this is always a good plot to have
    g.map_diag(sns.kdeplot, lw=3, legend=False)
    
    #df = df.loc[df['ni']!=0.027]
    fig = px.scatter_3d(df2, x="centroid_x",y="centroid_y",z="centroid_z",color="ni_log")
    fig.update_traces(marker_size=2)
    #fig.update_layout(font=dict(size=16))
    fig.update_layout(scene_aspectmode='data')  
    fig.update_scenes(xaxis = dict(title = 'X', tickfont = dict(size = 12), titlefont = dict(size = 18)))
    fig.update_scenes(yaxis = dict(title = 'Y', tickfont = dict(size = 12), titlefont = dict(size = 18)))
    fig.update_scenes(zaxis = dict(title = 'Z', tickfont = dict(size = 12), titlefont = dict(size = 18)))
    
    fields = ['centroid_x','centroid_y','centroid_z','ni',"co","mg","al","fe","si","domain","ore_type"]
    df3 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Murrin Murrin\\m08p06gc_model.csv", skipinitialspace=True,usecols=fields,low_memory=False)
    df3 = df3[3:]
    

    
    import skgstat as skg
    V = skg.Variogram(np.array(df[['centroid_x','centroid_y','centroid_z']]),np.array(df['ni']))
    V.plot()
    
    
    
    fields = ['centroid_x','centroid_y','centroid_z','ni',"co","mg","al","fe","si","domain","ore_type"]
    df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Murrin Murrin\\m08p06gc_model.csv", skipinitialspace=True,usecols=fields,low_memory=False)
    df = df[3:]
    df = df.astype(np.float16)
    #a = df[df.duplicated(subset=['centroid_x','centroid_y','centroid_z'],keep='first')]
    #a = df.groupby(by=['centroid_x','centroid_y','centroid_z']).size()
    # duplicaterow = df[df.duplicated(['centroid_x','centroid_y','centroid_z'],keep=False)]
    # row_no_duplicate = df.drop(duplicaterow.index)
    
    
    df_single = df.groupby(by=['centroid_x','centroid_y','centroid_z','domain','ore_type']).mean()
    df_single = df_single.reset_index()
    sub_df = df[(pd.to_numeric(df["centroid_x"], errors='coerce')==7064) & (pd.to_numeric(df["centroid_y"], errors='coerce')==19344)& (pd.to_numeric(df["centroid_z"], errors='coerce')==401)]
    
    
    fig = px.scatter_3d(df_single, x="centroid_x",y="centroid_y",z="centroid_z",color="ni_log")
    fig.update_traces(marker_size=2)
    #fig.update_layout(font=dict(size=16))
    fig.update_layout(scene_aspectmode='data')  
    fig.update_scenes(xaxis = dict(title = 'X', tickfont = dict(size = 12), titlefont = dict(size = 18)))
    fig.update_scenes(yaxis = dict(title = 'Y', tickfont = dict(size = 12), titlefont = dict(size = 18)))
    fig.update_scenes(zaxis = dict(title = 'Z', tickfont = dict(size = 12), titlefont = dict(size = 18)))


    plt.hist(np.log(df_single['ni']),bins=np.arange(-4,4,0.1),color='b')
    plt.xlabel('log Ni',fontsize=14)
    plt.ylabel('frequency',fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.title('histogram of log Ni in Murrin Murrin',fontsize=14)
    
    df_single2 = df_single.loc[df_single['ni']!=0.027]
    plt.hist(np.log(df_single2['ni']),bins=np.arange(-4,4,0.1),color='b')
    plt.xlabel('log Ni',fontsize=14)
    plt.ylabel('frequency',fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    fig = px.scatter_3d(df_single2, x="centroid_x",y="centroid_y",z="centroid_z",color="ni_log")
    fig.update_traces(marker_size=2)
    #fig.update_layout(font=dict(size=16))
    fig.update_layout(scene_aspectmode='data')  
    fig.update_scenes(xaxis = dict(title = 'X', tickfont = dict(size = 12), titlefont = dict(size = 18)))
    fig.update_scenes(yaxis = dict(title = 'Y', tickfont = dict(size = 12), titlefont = dict(size = 18)))
    fig.update_scenes(zaxis = dict(title = 'Z', tickfont = dict(size = 12), titlefont = dict(size = 18)))
    
    
    sub_df = df_single2[(pd.to_numeric(df_single2["centroid_x"], errors='coerce')==7064) & (pd.to_numeric(df_single2["centroid_y"], errors='coerce')==19344)]
    sub_df = sub_df.reset_index(drop=True)
    sub_df = sub_df.sort_values(by=['centroid_z'])
    
    sub_df1 = df_single2[(pd.to_numeric(df_single2["centroid_x"], errors='coerce')==7480) & (pd.to_numeric(df_single2["centroid_y"], errors='coerce')==19584)]
    sub_df1 = sub_df1.reset_index(drop=True)
    sub_df1 = sub_df1.sort_values(by=['centroid_z'])
    
    sub_df2 = df_single2[(pd.to_numeric(df_single2["centroid_x"], errors='coerce')==7400) & (pd.to_numeric(df_single2["centroid_y"], errors='coerce')==19472)]
    sub_df2 = sub_df2.reset_index(drop=True)
    sub_df2 = sub_df2.sort_values(by=['centroid_z'])
    
        
    sub_df3 = df_single2[(pd.to_numeric(df_single2["centroid_x"], errors='coerce')==7164) & (pd.to_numeric(df_single2["centroid_y"], errors='coerce')==19264)]
    sub_df3 = sub_df3.reset_index(drop=True)
    sub_df3 = sub_df3.sort_values(by=['centroid_z'])
    
    sub_df4 = df_single2[(pd.to_numeric(df_single2["centroid_x"], errors='coerce')==7560) & (pd.to_numeric(df_single2["centroid_y"], errors='coerce')==19184)]
    sub_df4 = sub_df4.reset_index(drop=True)
    sub_df4 = sub_df4.sort_values(by=['centroid_z'])
    
    plt.figure(figsize=(12,6))
    plt.plot(sub_df['centroid_z'],sub_df['ni_log'],color='b',label='bore core 1')
    plt.plot(sub_df1['centroid_z'],sub_df1['ni_log'],color='g',label='bore core 2')
    plt.plot(sub_df2['centroid_z'],sub_df2['ni_log'],color='r',label='bore core 3')
    plt.plot(sub_df3['centroid_z'],sub_df3['ni_log'],color='m',label='bore core 4')
    plt.plot(sub_df3['centroid_z'],sub_df3['ni_log'],color='k',label='bore core 5')
    plt.xlabel('depth of bore hole',fontsize=14)
    plt.ylabel('Ni grade',fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    
    df_single3 = df_single2[['ni_log','co_log','mg_log','al_log','fe_log','si_log']]
    g = sns.PairGrid(df_single3)
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot) # this is always a good plot to have
    g.map_diag(sns.kdeplot, lw=3, legend=False)
    
    df_single2['domain'] = pd.to_numeric(df_single2['domain'])
    df_single2.groupby(['domain']).size().plot(kind='bar',color='b')
    plt.xlabel('geological domain')
    plt.ylabel('frequency')
    
    df_single2['ore_type'] = pd.to_numeric(df_single2['ore_type'])
    df_single2.groupby(['ore_type']).size().plot(kind='bar',color='b')
    plt.xlabel('ore tpye')
    plt.ylabel('frequency')
    ##power law##
    # df = df.sample(frac=1)
    # df = df.reset_index(drop=True)
    # df5 = np.array(df['ni'])
    # variance = []
    # list_mean_subgroup = []
    # for j in range(1,500,1):
    #     N=j
    #     subgroup = [df5[n:n+N] for n in range(0,len(df5),N)]
    #     mean_subgroup = []
    #     for i in subgroup:
    #         mean_subgroup.append(i.mean())
    #     list_mean_subgroup.append(mean_subgroup)    
    #     variance_N = np.var(mean_subgroup)
    #     variance.append(variance_N)
    # variance = np.array(variance)
    # interval = np.arange(1,500,1)
    # variance_log = np.log10(variance)
    # interval_log = np.log10(interval)  
    
    # mean_subgroup_df = pd.DataFrame(list_mean_subgroup)
    # mean_subgroup_df = pd.DataFrame.transpose(mean_subgroup_df)
    # #mean_subgroup_df.columns = ['n=1','n=6','n=11','n=16','n=21','n=26']
    # import matplotlib.pyplot as plt
    # plt.scatter(interval_log,variance_log)
    # plt.xlabel('log(N)')
    # plt.ylabel('log(variance) of Ni grade')
    
    
    
    
    
    
    
    
    
    