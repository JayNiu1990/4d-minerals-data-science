if __name__ == '__main__':
    import warnings
    import matplotlib.pyplot as plt
    import plotly.express as px
    import numpy as np
    import pandas as pd    
    import plotly.io as pio
    import plotly.graph_objs as go
    import seaborn as sns
    fields = ['BHID','Fe_dh','As_dh','CuT_dh',"X","Y","Z","LITH"]
    pio.renderers.default='browser'
    df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)
    df = df.dropna()
    df = df[(pd.to_numeric(df["CuT_dh"], errors='coerce')>0) & (pd.to_numeric(df["Fe_dh"], errors='coerce')>0)& (pd.to_numeric(df["As_dh"], errors='coerce')>0)
            & (pd.to_numeric(df["X"], errors='coerce')>=17000)& (pd.to_numeric(df["X"], errors='coerce')<17500)
            & (pd.to_numeric(df["Y"], errors='coerce')>=107000)& (pd.to_numeric(df["Y"], errors='coerce')<107500)
            & (pd.to_numeric(df["Z"], errors='coerce')>=2500)& (pd.to_numeric(df["Z"], errors='coerce')<3000)]
    fig = px.scatter_3d(df, x="X",y="Y",z="Z",color="BHID")
    fig.update_traces(marker_size=2)
    fig.update_layout(font=dict(size=10))
    fig.update_layout(scene_aspectmode='data')
    fig.show()    
    
    df['LITH'] = df['LITH'].astype(int)
    df = df.reset_index(drop=True)
    df["CuT_dh"] = df["CuT_dh"].astype("float")
    df["Fe_dh"] = df["Fe_dh"].astype("float")
    df["As_dh"] = df["As_dh"].astype("float")
    # add gaussian noise
    df['X'] = round(df['X'],2)
    df['Y'] = round(df['Y'],2)
    df['Z'] = round(df['Z'],2)
    mu, sigma = 0.1, 0.01
    
    np.random.seed(0)
    noise = pd.DataFrame(np.random.normal(mu, sigma, [len(df),1])) 
    
    noise = round(noise,2)
    noise.columns = ['noise']
    
    df['CuT_dh_log'] = np.log(df['CuT_dh'])
    df['CuT_dh_log'] = round(df['CuT_dh_log'],3)
    
    df['Fe_dh_log'] = np.log(df['Fe_dh'])
    df['Fe_dh_log'] = round(df['Fe_dh_log'],3)
    
    df['As_dh_log'] = np.log(df['As_dh'])
    df['As_dh_log'] = round(df['As_dh_log'],3)
    
    df_new1 = pd.concat([df['CuT_dh_log'],noise['noise']],axis=1)
    df_new1['CuT_dh_log_noise'] = df_new1.sum(axis=1)
    df = pd.concat([df,df_new1],axis=1)
    
    df_new2 = pd.concat([df['Fe_dh_log'],noise['noise']],axis=1)
    df_new2['Fe_dh_log_noise'] = df_new2.sum(axis=1)
    df = pd.concat([df,df_new2],axis=1)
    
    df_new3 = pd.concat([df['As_dh_log'],noise['noise']],axis=1)
    df_new3['As_dh_log_noise'] = df_new3.sum(axis=1)
    df = pd.concat([df,df_new3],axis=1)
    
    df2 = df[['BHID','X','Y','Z','CuT_dh','Fe_dh','As_dh','CuT_dh_log_noise','Fe_dh_log_noise','As_dh_log_noise','LITH']]
    df2 = df2.reset_index(drop=True)


    # plt.figure(figsize=(20, 5))        
    # plt.scatter(df['LITH'],df['CuT_dh_log'])    
    # plt.axhline(y=df['CuT_dh_log'].mean())
    # plt.xticks(np.arange(0, 20, step=1))
    
    
    
    
    
    
    
    
    
    
    
    
    
    df_list1= []
    #######label elevation
    import pymc3 as pm
    for i in range(1200,3500,20):
        left = i
        right=i+20
        string1 = str(i) + '-' + str(i+20)
        df3 = df2.loc[(df2['Z']<i+20) & (df2['Z']>=i)]
        df3["elevation"] = pd.Series(string1,index=df2.loc[(df2['Z']<i+20) & (df2['Z']>=i)].index)
        df_list1.append(df3)
    df2_new = pd.concat(df_list1)
    df2_new = df2_new.reset_index(drop=True)
    

    
    elevation_idxs, elevation = pd.factorize(df2_new.elevation)
    df2_new['elevation_inxs'] = elevation_idxs
    n_elevation = len(df2_new.elevation.unique())
    
    plt.hist(df2_new['elevation_inxs'],bins=30)
    
    #df2_new = df2_new.sample(n=40000,random_state=10)
    df2_train = df2_new.sample(n=8000,random_state=10)
    df2_test = df2_new.drop(df2_train.index)
    df2_train = df2_train.sort_values(by=['elevation'])
    df2_test = df2_test.sort_values(by=['elevation'])
    # df2_train = df2_train.reset_index(drop=True)
    # df2_test = df2_test.reset_index(drop=True)
    
    elevation_idxs1 = np.array(df2_train['elevation_inxs'])
    elevation_idxs2 = np.array(df2_test['elevation_inxs'])
    
    
    fig = px.scatter_3d(df2_train, x="X",y="Y",z="Z",color="BHID")
    fig.update_traces(marker_size=2)
    fig.update_layout(font=dict(size=10))
    fig.update_layout(scene_aspectmode='data')
    fig.show()    
    
    fig = px.scatter_3d(df2_test, x="X",y="Y",z="Z",color="BHID")
    fig.update_traces(marker_size=2)
    fig.update_layout(font=dict(size=10))
    fig.update_layout(scene_aspectmode='data')
    fig.show()    
    with pm.Model() as pooled_model:
        alpha = pm.Normal('alpha',mu = 0, sd = 1)
        beta = pm.Normal('beta',mu = 0, sd = 1)
        eps = pm.Uniform('eps', lower=0, upper=1)
        Fe_mean = alpha + beta*df2_train['CuT_dh_log_noise'].values
        Fe = pm.Normal('Fe', mu = Fe_mean, sd = eps, observed = df2_train['Fe_dh_log_noise'])
    with pooled_model:
        pooled_trace = pm.sample(2000)
    pm.traceplot(pooled_trace)
    
    with pm.Model() as unpooled_model:
        #mu_alpha = pm.Normal('mu_alpha',mu=0,sd=100)
        #sigma_alpha = pm.HalfCauchy('sigma_alpha',2)
        
        #mu_beta = pm.Normal('mu_beta',mu=0,sd=100)
        #sigma_beta = pm.HalfCauchy('sigma_beta',2)
        
        #eps = pm.Uniform('eps',lower=0, upper=100)
        eps = pm.Uniform('eps', lower=0, upper=1)
        alpha = pm.Normal('alpha',mu = 0,sd = 1,shape = n_elevation)
        beta = pm.Normal('beta',mu = 0,sd = 1, shape = n_elevation)
        
        Fe_mean = alpha[elevation_idxs1] + beta[elevation_idxs1]*df2_train['CuT_dh_log_noise'].values
        Fe = pm.Normal('Fe', mu = Fe_mean, sd = eps,observed = df2_train['Fe_dh_log_noise'])
        
    with unpooled_model:
        unpooled_trace = pm.sample(2000)

    
    with pm.Model() as hirearchical_model:
        mu_alpha = pm.Normal('mu_alpha',mu=0,sd=1)
        sigma_alpha = pm.Uniform('sigma_alpha', lower=0, upper=1)
        
        mu_beta = pm.Normal('mu_beta',mu=0,sd=1)
        sigma_beta = pm.Uniform('sigma_beta', lower=0, upper=1)
        
        eps = pm.Uniform('eps', lower=0, upper=1)
        
        alpha = pm.Normal('alpha',mu = mu_alpha,sd = sigma_alpha, shape = n_elevation)
        beta = pm.Normal('beta',mu = mu_beta,sd = sigma_beta, shape = n_elevation)
        
        Fe_mean = alpha[elevation_idxs1] + beta[elevation_idxs1]*df2_train['CuT_dh_log_noise'].values
        Fe = pm.Normal('Fe', mu = Fe_mean, sd = eps,observed = df2_train['Fe_dh_log_noise'], shape = n_elevation)
        
    with hirearchical_model:
        hirearchical_trace = pm.sample(2000)    

    import arviz as az
    a = az.summary(unpooled_trace)
    az.plot_trace(pooled_trace,figsize=(14,14))
    az.plot_trace(unpooled_trace,figsize=(14,14))
    az.plot_trace(hirearchical_trace,figsize=(14,14))

    # selection = list(df2_train.elevation.unique())[1:7] 
    # fig,axis = plt.subplots(3,2,figsize=(12,9),sharey=True,sharex=True);
    # axis = axis.ravel()
    # for i,c in enumerate(selection):
    #     c_data = df2_train.loc[df2_train.elevation==c]
    #     c_data = c_data.reset_index(drop=True)
    #     c_index = np.where(elevation==c)[0][0]
    #     xvals = np.linspace(-3,3)
    #     # for a_val, b_val in zip(hirearchical_trace['alpha'][1000:,c_index],hirearchical_trace['beta'][1000:,c_index]):
    #     #     axis[i].plot(xvals,a_val+b_val*xvals,'b',alpha=.005)
    #     #axis[i].plot(xvals,hirearchical_trace['alpha'][1000:,c_index].mean()+hirearchical_trace['beta'][1000:,c_index].mean()*xvals,'r',alpha=1,lw=1.,label='hirearchical')
    #     for a_val, b_val in zip(unpooled_trace['alpha'][1000:,c_index],unpooled_trace['beta'][1000:,c_index]):
    #         axis[i].plot(xvals,a_val+b_val*xvals,'g',alpha=.1)
    #     #axis[i].plot(xvals,unpooled_trace['alpha'][1000:,c_index].mean()+unpooled_trace['beta'][1000:,c_index].mean()*xvals,'g',alpha=1,lw=1.,label='individual')
    #     axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=80,label='original data')
    #     axis[i].set_xlim(-3,3)
    #     axis[i].set_ylim(-5,5)
    #     # axis[i].set_title(c)
    #     if not i%6:
    #         axis[i].legend(fontsize=10)
            
            
    hier_a = hirearchical_trace['alpha'][:].mean(axis=0)
    hier_b = hirearchical_trace['beta'][:].mean(axis=0)
    indv_a = unpooled_trace['alpha'][:].mean(axis=0)
    indv_b = unpooled_trace['beta'][:].mean(axis=0)
    compl_a = pooled_trace['alpha'][:].mean(axis=0)
    compl_b = pooled_trace['beta'][:].mean(axis=0)
    
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, xlabel='alpha', ylabel='beta', 
                         title='Hierarchical vs. Non-hierarchical Bayes', 
                         xlim=(0, 2.5), ylim=(-0.75,0.75))
    
    ax.scatter(indv_a,indv_b,c='c', s=26, alpha=1, label = 'non-pooling')
    ax.scatter(hier_a,hier_b, c='r', s=26, alpha=1, label = 'partial pooling')
    ax.scatter(compl_a,compl_b,c='m',marker='o', s=200, alpha=1, label = 'complete pooling')
    ax.legend(fontsize=16);
    ax.set_xlabel('alpha',fontsize=30)
    ax.set_ylabel('beta',fontsize=30)
    ax.axhline(y=0,color='k',linewidth=5)
    for i in range(len(indv_b)):  
        ax.arrow(indv_a[i], indv_b[i], hier_a[i] - indv_a[i], hier_b[i] - indv_b[i], 
                 fc="k", ec="k", length_includes_head=True, alpha= 1 , head_width=.02)    
        
    #plt.scatter(df2['CuT_dh_log_noise'],df2['Fe_dh_log_noise'])
    

    # selection1 = list(df2_test.elevation.unique())[1:7] #['2000-2100','2100-2200', '2200-2300'] 
    # fig,axis = plt.subplots(3,2,figsize=(12,9),sharey=True,sharex=True);
    # axis = axis.ravel()
    # for i,c in enumerate(selection1):
    #     c_data = df2_test.loc[df2_test.elevation==c]
    #     c_data = c_data.reset_index(drop=True)
    #     c_index = np.where(elevation==c)[0][0]
    #     z = list(c_data['elevation_inxs'])[0]
    #     xvals = np.linspace(-5,3)
    #     for a_val, b_val in zip(hirearchical_trace['alpha'][1000:,c_index],hirearchical_trace['beta'][1000:,c_index]):
    #         axis[i].plot(xvals,a_val+b_val*xvals,'b',alpha=.005)
    #     #axis[i].plot(xvals,hirearchical_trace['alpha'][1000:,c_index].mean()+hirearchical_trace['beta'][1000:,c_index].mean()*xvals,'r',alpha=1,lw=1.,label='hirearchical')
    #     # for a_val, b_val in zip(unpooled_trace['alpha'][1000:,c_index],unpooled_trace['beta'][1000:,c_index]):
    #     #     axis[i].plot(xvals,a_val+b_val*xvals,'g',alpha=.1)
    #     #axis[i].plot(xvals,unpooled_trace['alpha'][1000:,c_index].mean()+unpooled_trace['beta'][1000:,c_index].mean()*xvals,'g',alpha=1,lw=1.,label='individual')
    #     axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=80,label='original data')
    #     axis[i].set_xlim(-3,3)
    #     axis[i].set_ylim(-5,5)
    #     # axis[i].set_title(c)
    #     #if not i%6:
    #         #axis[i].legend(fontsize=10)        

    # for i in np.linspace(-1,3):
    #     print(hirearchical_trace['alpha'][:,0]+hirearchical_trace['beta'][:,0]*i)
    # import seaborn as sns
    # sns.regplot(
    #     df_new[df_new['elevation']=='2000-2100']['Fe_dh_log_noise'],
    #     df_new[df_new['elevation']=='2000-2100']['CuT_dh_log_noise'],
    #     scatter_kws={"color": "b"},
    #     line_kws={"color": "k"},
    # )
    #################error#################
    n=93
    seq1 = np.linspace(-5,3,20) 
    pred1 = np.zeros((len(seq1),len(pooled_trace[:][::10])*pooled_trace.nchains))
    for i, w in enumerate(seq1):
        pred1[i] = pooled_trace[:][::10]['alpha'] + pooled_trace[:][::10]['beta']*w
    seq2 = np.linspace(-5,3,20) 
    pred2 = np.zeros((len(seq2),len(unpooled_trace[:][::10])*unpooled_trace.nchains))
    for i, w in enumerate(seq2):
        pred2[i] = unpooled_trace[:][::10]['alpha'][:,n] + unpooled_trace[:][::10]['beta'][:,n]*w
        
        
    seq3 = np.linspace(-5,3,20) 
    pred3 = np.zeros((len(seq3),len(hirearchical_trace[:][::10])*hirearchical_trace.nchains))
    for i, w in enumerate(seq3):
        pred3[i] = hirearchical_trace[:][::10]['alpha'][:,n] + hirearchical_trace[:][::10]['beta'][:,n]*w

    #plt.plot(seq1, pred1, '.',color='r')
    
    ##################error bar################
    plt.figure(figsize=(12, 8))
    x1 = np.linspace(-5,5,20) 
    y1 = pred1.mean(axis=1)
    y_min1 = pred1.mean(axis=1) - pred1.min(axis=1)
    y_max1 = pred1.max(axis=1) - pred1.mean(axis=1)
    yerr = np.vstack((y_min1,y_max1))
    plt.errorbar(x1,y1,yerr=yerr,fmt='o',color='c',alpha=0.5,label='complete pooling')
    
    x2 = np.linspace(-5,5,20) 
    y2 = pred2.mean(axis=1)
    y_min2 = pred2.mean(axis=1) - pred2.min(axis=1)
    y_max2 = pred2.max(axis=1) - pred2.mean(axis=1)
    yerr2 = np.vstack((y_min2,y_max2))
    plt.errorbar(x2,y2,yerr=yerr2,fmt='o',color='b',alpha=0.5,label='no pooling')
    
    x3 = np.linspace(-5,5,20) 
    y3 = pred3.mean(axis=1)
    y_min3 = pred3.mean(axis=1) - pred3.min(axis=1)
    y_max3 = pred3.max(axis=1) - pred3.mean(axis=1)
    yerr3 = np.vstack((y_min3,y_max3))
    plt.errorbar(x3,y3,yerr=yerr3,fmt='o',color='r',alpha=0.5,label='partial pooling (hierarchical)')
    plt.scatter(df2_train[df2_train['elevation_inxs']==n]['CuT_dh_log_noise'], df2_train[df2_train['elevation_inxs']==n]['Fe_dh_log_noise'],label='bore core (training)',s=200,color='k',marker = '.')
    plt.scatter(df2_test[df2_test['elevation_inxs']==n]['CuT_dh_log_noise'], df2_test[df2_test['elevation_inxs']==n]['Fe_dh_log_noise'],label='bore core (testing)',s=100,color='m',marker = 'v')
    plt.legend(fontsize=14,loc='upper left')
    plt.xlabel('log Cu')
    plt.ylabel('log Fe')
    plt.ylim(-5,5)
    plt.title('Elevation:' + str(df2_train[df2_train['elevation_inxs']==n]['elevation'].unique()[0]))
    plt.show()







    #plt.scatter(df2_train[df2_train['elevation_inxs']==1]['CuT_dh_log_noise'], df2_train[df2_train['elevation_inxs']==1]['Fe_dh_log_noise'],label='bore core')
    #plt.plot(seq1, pred1.mean(1), 'k')
    # az.plot_hdi(seq1, pred1.T,color='c')
    # az.plot_hdi(seq2, pred2.T,color='r')
    # az.plot_hdi(seq3, pred3.T,color='g')
    # plt.legend()
    
    seq = np.linspace(-3,3,20) 
    pred = np.zeros((len(seq),len(unpooled_trace[:][::10])*unpooled_trace.nchains))
    for i, w in enumerate(seq):
        pred[i] = unpooled_trace[:][::10]['alpha'][:,1] + unpooled_trace[:][::10]['beta'][:,1]*w
    plt.scatter(df2_test[df2_test['elevation_inxs']==1]['CuT_dh_log_noise'], df2_test[df2_test['elevation_inxs']==1]['Fe_dh_log_noise'])
    plt.plot(seq, pred.mean(1), 'k')
    az.plot_hdi(seq, pred.T)

    seq1 = np.linspace(-3,3,20) 
    pred1= np.zeros((len(seq1),len(hirearchical_trace[:][::10])*hirearchical_trace.nchains))
    for i, w in enumerate(seq1):
        pred1[i] = hirearchical_trace[:][::10]['alpha'][:,1] + hirearchical_trace[:][::10]['beta'][:,1]*w
    plt.scatter(df2_test[df2_test['elevation_inxs']==1]['CuT_dh_log_noise'], df2_test[df2_test['elevation_inxs']==1]['Fe_dh_log_noise'])
    plt.plot(seq1, pred1.mean(1), 'k')
    az.plot_hdi(seq, pred.T)
    az.plot_hdi(seq1, pred1.T)
    
    df_test_label = [val for val in list(df2_test.elevation.unique()) for _ in (0, 1)]

    df_label = [val for val in list(df2_train.elevation.unique()) for _ in (0, 1)]

    selection2 = list(df2_train.elevation.unique())[85:88] 
    # #selection1 = ['2560-2580','2580-2600','2600-2620','2620-2640','2640-2660','2660-2680']  #[1:4] #['2000-2100','2100-2200', '2200-2300'] 
    fig,axis = plt.subplots(3,1,figsize=(8,12),sharey=True,sharex=True);
    axis = axis.ravel()
    for i,c in enumerate(selection2):
        #if i%2==1:
        c_data = df2_test.loc[df2_test.elevation==c]
        c_data = c_data.reset_index(drop=True)
        c_index = np.where(elevation==c)[0][0]
        
        c_data1 = df2_train.loc[df2_train.elevation==c]
        c_data1 = c_data1.reset_index(drop=True)
        c_index = np.where(elevation==c)[0][0]
        
        xvals = np.linspace(-5,3)
        axis[i].set_title('elevation:' + str(c))
        
        axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=200,label = 'bore core (training)')
        axis[i].scatter(c_data1['CuT_dh_log_noise'],c_data1['Fe_dh_log_noise'],color='m',marker='v',s=100,label = 'bore core(testing)')
        axis[i].plot(xvals,hirearchical_trace['alpha'][1000:,c_index].mean()+hirearchical_trace['beta'][1000:,c_index].mean()*xvals,'c',alpha=1,lw=3.,label='partial pooling')
        axis[i].plot(xvals,unpooled_trace['alpha'][1000:,c_index].mean()+unpooled_trace['beta'][1000:,c_index].mean()*xvals,'r',alpha=1,lw=3.,label='no pooling')
        axis[i].plot(xvals,pooled_trace['alpha'][1000:].mean()+pooled_trace['beta'][1000:].mean()*xvals,'b',alpha=1,lw=3.,label='complete pooling')
        
        axis[i].set_xlim(-3,3)
        axis[i].set_ylim(-5,5)
        axis[i].legend(loc='lower right',prop={'size':12})
        axis[i].set_xlabel('log Cu')
        axis[i].set_ylabel('log Fe')
        # else:
        #     c_data = df2_train.loc[df2_train.elevation==c]
        #     c_data = c_data.reset_index(drop=True)
        #     c_index = np.where(elevation==c)[0][0]
        #     xvals = np.linspace(-5,3)
        #     axis[i].set_title('elevation:' + str(c) + '(training)')
            
        #     axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=200,label = 'bore core')
        #     axis[i].plot(xvals,hirearchical_trace['alpha'][1000:,c_index].mean()+hirearchical_trace['beta'][1000:,c_index].mean()*xvals,'c',alpha=1,lw=3.,label='partial pooling')
        #     axis[i].plot(xvals,unpooled_trace['alpha'][1000:,c_index].mean()+unpooled_trace['beta'][1000:,c_index].mean()*xvals,'r',alpha=1,lw=3.,label='no pooling')
        #     axis[i].plot(xvals,pooled_trace['alpha'][1000:].mean()+pooled_trace['beta'][1000:].mean()*xvals,'b',alpha=1,lw=3.,label='complete pooling')
            
        #     axis[i].set_xlim(-3,3)
        #     axis[i].set_ylim(-5,5)
        #     axis[i].legend(loc='lower right',prop={'size':16})
        #     axis[i].set_xlabel('log Cu')
        #     axis[i].set_ylabel('log Fe')

    df_test_label1 = [val for val in df_test_label for _ in (0, 1)]
    df_test_label1 = pd.DataFrame(df_test_label1,columns=['elevation'])
    df_test_label1_drop = df_test_label1.drop_duplicates(keep='first')
    df_test_label2 = df_test_label1.drop(df_test_label1_drop.index)['elevation'].to_list()
    
    
    df_train_label = [val for val in list(df2_train.elevation.unique()) for _ in (0, 1)]
    df_train_label1 = [val for val in df_train_label for _ in (0, 1)]
    df_train_label1 = pd.DataFrame(df_train_label1,columns=['elevation'])
    df_train_label1_drop = df_train_label1.drop_duplicates(keep='first')
    df_train_label2 = df_train_label1.drop(df_train_label1_drop.index)['elevation'].to_list()
    
    ###################plot complete/no/partial pooling regression on training/testing data####################
    selection3 = df_train_label2[255:264] #list(df2_test.elevation.unique())[1:4] #['2000-2100','2100-2200', '2200-2300'] 
    fig,axis = plt.subplots(3,3,figsize=(15,12),sharey=True,sharex=True);
    axis = axis.ravel()
    for i,c in enumerate(selection3):
        c_data = df2_train.loc[df2_train.elevation==c]
        c_data = c_data.reset_index(drop=True)
        c_index = np.where(elevation==c)[0][0]
        c_data1 = df2_test.loc[df2_test.elevation==c]
        c_data1 = c_data1.reset_index(drop=True)
        c_index1 = np.where(elevation==c)[0][0]
        xvals = np.linspace(-5,3)
        if (i-1)%3 ==1:#258
            axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=200,label = 'bore core (training)')
            axis[i].scatter(c_data1['CuT_dh_log_noise'],c_data1['Fe_dh_log_noise'],color='m',marker='v',s=100,label = 'bore core(testing)')
            axis[i].set_title('elevation:' + str(c))
            num=1
            for a_val, b_val in zip(hirearchical_trace['alpha'][1000:,c_index],hirearchical_trace['beta'][1000:,c_index]):
                if num==1:
                    axis[i].plot(xvals,a_val+b_val*xvals,'c',label = 'partial pooling')
                    num+=1
                else:
                    axis[i].plot(xvals,a_val+b_val*xvals,'c',alpha=.003)
        elif (i+1)%3 ==2: #147
            axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=200,label = 'bore core (training)')  
            axis[i].scatter(c_data1['CuT_dh_log_noise'],c_data1['Fe_dh_log_noise'],color='m',marker='v',s=100,label = 'bore core(testing)')
            axis[i].set_title('elevation:' + str(c))
            num=1
        #axis[i].plot(xvals,hirearchical_trace['alpha'][1000:,c_index].mean()+hirearchical_trace['beta'][1000:,c_index].mean()*xvals,'r',alpha=1,lw=1.,label='hirearchical')
            for a_val, b_val in zip(unpooled_trace['alpha'][1000:,c_index],unpooled_trace['beta'][1000:,c_index]):  
                if num==1:
                    axis[i].plot(xvals,a_val+b_val*xvals,'g',label = 'no pooling')
                    num+=1
                else:
                    axis[i].plot(xvals,a_val+b_val*xvals,'g',alpha=.003)
        else:
            axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=200,label = 'bore core (training)')  
            axis[i].scatter(c_data1['CuT_dh_log_noise'],c_data1['Fe_dh_log_noise'],color='m',marker='v',s=100,label = 'bore core(testing)')
            axis[i].set_title('elevation:' + str(c))
            num=1
            for a_val, b_val in zip(pooled_trace['alpha'][1000:],pooled_trace['beta'][1000:]):  
                if num==1:
                    axis[i].plot(xvals,a_val+b_val*xvals,'b',label = 'complete pooling')
                    num+=1
                else:
                    axis[i].plot(xvals,a_val+b_val*xvals,'b',alpha=.003)
        #axis[i].plot(xvals,unpooled_trace['alpha'][1000:,c_index].mean()+unpooled_trace['beta'][1000:,c_index].mean()*xvals,'g',alpha=1,lw=1.,label='individual')
        axis[i].set_xlim(-3,3)
        axis[i].set_ylim(-5,5)
        axis[i].legend(fontsize=12,loc='lower right')
        axis[i].set_xlabel('log Cu')
        axis[i].set_ylabel('log Fe')
        
    
    ###################plot complete/no/partial pooling regression on testing data####################
    selection2 = df_test_label2[201:210]
    fig,axis = plt.subplots(3,3,figsize=(15,12),sharey=True,sharex=True);
    axis = axis.ravel()
    for i,c in enumerate(selection2):
        c_data = df2_test.loc[df2_test.elevation==c]
        c_data = c_data.reset_index(drop=True)
        c_index = np.where(elevation==c)[0][0]
        z = list(c_data['elevation_inxs'])[0]
        xvals = np.linspace(-5,3)
        if (i-1)%3 ==1:#258
            axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=200,label = 'bore core')
            axis[i].set_title('elevation:' + str(c))
            num=1
            for a_val, b_val in zip(hirearchical_trace['alpha'][1000:,c_index],hirearchical_trace['beta'][1000:,c_index]):
                if num==1:
                    axis[i].plot(xvals,a_val+b_val*xvals,'c',label = 'partial pooling')
                    num+=1
                else:
                    axis[i].plot(xvals,a_val+b_val*xvals,'c',alpha=.01)
        elif (i+1)%3 ==2: #147
            axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=200,label = 'bore core')  
            axis[i].set_title('elevation:' + str(c))
            num=1
        #axis[i].plot(xvals,hirearchical_trace['alpha'][1000:,c_index].mean()+hirearchical_trace['beta'][1000:,c_index].mean()*xvals,'r',alpha=1,lw=1.,label='hirearchical')
            for a_val, b_val in zip(unpooled_trace['alpha'][1000:,c_index],unpooled_trace['beta'][1000:,c_index]):  
                if num==1:
                    axis[i].plot(xvals,a_val+b_val*xvals,'r',label = 'no pooling')
                    num+=1
                else:
                    axis[i].plot(xvals,a_val+b_val*xvals,'r',alpha=.01)
        else:
            axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=200,label = 'bore core')  
            axis[i].set_title('elevation:' + str(c))
            num=1
            for a_val, b_val in zip(pooled_trace['alpha'][1000:],pooled_trace['beta'][1000:]):  
                if num==1:
                    axis[i].plot(xvals,a_val+b_val*xvals,'b',label = 'complete pooling')
                    num+=1
                else:
                    axis[i].plot(xvals,a_val+b_val*xvals,'b',alpha=.01)
        #axis[i].plot(xvals,unpooled_trace['alpha'][1000:,c_index].mean()+unpooled_trace['beta'][1000:,c_index].mean()*xvals,'g',alpha=1,lw=1.,label='individual')
        axis[i].set_xlim(-3,3)
        axis[i].set_ylim(-5,5)
        axis[i].legend(loc='lower right')
        axis[i].set_xlabel('log Cu')
        axis[i].set_ylabel('log Fe')
        

    
    
    