if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import math
    import plotly.express as px
    import numpy as np
    import pandas as pd    
    import matplotlib.pyplot as plt
    import plotly.io as pio
    import plotly.graph_objs as go
    pio.renderers.default='browser'
    with open('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Cadia East\\CE_Collarsmod.txt') as f:
        lines1 = f.readlines()
    list1 = []
    for line1 in lines1[1:]:
        line = line1.split()
        row = np.array(line[0:12])
        list1.append(row)
    data1 = pd.DataFrame(list1,columns=['NAME','REGION','DRILLHOLE','X','Y','Z','DEPTH','DATE1','DATE2','D','AZIMUTH','DIP'])
    
    
    str_list = ["UE035","UE041","UE040","UE055","UE054","UE056","UE100","UE101","UE099",
     "UE102","UE051","UE049","UE050","UE048","UE047","UE103","UE097","UE104",
     "UE096","UE018","UE017","UE042","UE043","UE044","UE045","UE046","UE092",
     "UE095","UE113","UE090","UE091A","UE094","UE013","UE011","UE009","UE010",
     "UE036","UE019A","UE037","UE020","UE022","UE021","UE023","UE024","UE025",
     "UE026","UE027","UE028","UE029","UE014","UE012","UE015"]
    str_list.sort()
    
    data_list = []
    for _ in str_list[0:46]:
        str1 = _
        AZIMUTH = list(data1[data1['NAME']==str1]['AZIMUTH'])[0].astype('float64')
        DIP = list(data1[data1['NAME']==str1]['DIP'])[0].astype('float64')
        X = list(data1[data1['NAME']==str1]['X'])[0].astype('float64')
        Y = list(data1[data1['NAME']==str1]['Y'])[0].astype('float64')
        Z = list(data1[data1['NAME']==str1]['Z'])[0].astype('float64')
        
        with open('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Cadia East\\UE001toUE099.txt') as f:
            lines2 = f.readlines()
        
        list2 = []
        for line2 in lines2[1:]:
            line = line2.split()
            row = np.concatenate((np.array(line[0:6]),np.array(line[11:12])))
            list2.append(row)
            
        data2 = pd.DataFrame(list2,columns=['SAMPLE','HOLEID','PROJECTCODE','FROM','TO','AU_ppm','CU_ppm'])
        data2 = data2.dropna()
        data2 = data2[data2['HOLEID']==str1]
        data_list.append(data2)
        data2['X'] = round(X + ((data2['FROM'].astype('float64')+data2['TO'].astype('float64'))*0.5 * math.sin(math.radians(AZIMUTH)) * math.cos(math.radians(DIP))),3)
        data2['Y'] = round(Y + ((data2['FROM'].astype('float64')+data2['TO'].astype('float64'))*0.5 * math.cos(math.radians(AZIMUTH)) * math.cos(math.radians(DIP))),3)
        data2['Z'] = round(Z + ((data2['FROM'].astype('float64')+data2['TO'].astype('float64'))*0.5 * math.sin(math.radians(DIP))),3)
    data = pd.concat(data_list)
    
    data = data[(data['HOLEID']!='UE011') & (data['HOLEID']!='UE010')& (data['HOLEID']!='UE009')]
    
    data = data[(pd.to_numeric(data["AU_ppm"], errors='coerce')>0) & (pd.to_numeric(data["CU_ppm"], errors='coerce')>0)]
    data = data.reset_index(drop=True) 
    
    
    data['AU_ppm'] = data['AU_ppm'].astype('float')
    data['CU_ppm'] = data['CU_ppm'].astype('float')
    data['CU_wt'] = data['CU_ppm']/10000
    data['log Cu_wt'] = np.log(data['CU_wt'])
    data['log AU_ppm'] = np.log(data['AU_ppm'])
    data.to_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Cadia East\\Cadia_East_pc2.csv')  

    mu, sigma = 0.1, 0.01
    
    np.random.seed(0)
    
    noise = pd.DataFrame(np.random.normal(mu, sigma, [len(data),1])) 
    noise = round(noise,3)
    noise.columns = ['noise']
    
    data['AU_ppm_log'] = np.log(data['AU_ppm'])
    data['AU_ppm_log'] = round(data['AU_ppm_log'],3)
    
    data['CU_wt_log'] = np.log(data['CU_wt'])
    data['CU_wt_log'] = round(data['CU_wt_log'],3)
    
    df_new1 = pd.concat([data['CU_wt_log'],noise['noise']],axis=1)
    df_new1['CU_wt_log_noise'] = df_new1.sum(axis=1)
    data = pd.concat([data,df_new1],axis=1)
    df_new2 = pd.concat([data['AU_ppm_log'],noise['noise']],axis=1)
    df_new2['AU_ppm_log_noise'] = df_new2.sum(axis=1)
    
    data = pd.concat([data,df_new2],axis=1)
    
    
    
    
    
    #plt.scatter(data['CU_wt_log'],data['AU_ppm_log'])
    
    # fig = px.scatter_3d(data, x="X",y="Y",z="Z",color="CU_ppm")
    # fig.update_traces(marker_size=2)
    # fig.update_layout(font=dict(size=14))
    # fig.show()    
    n = 100
    m = 100
    xx1 = np.arange(15200, 16000, n).astype('float64')
    yy1 = np.arange(21500, 22100, n).astype('float64')
    zz1 = np.arange(4400, 5300, m).astype('float64')
    
    blocks = []
    for k in zz1:
        for j in yy1:
            for i in xx1:
                sub_block = data.loc[(pd.to_numeric(data["X"], errors='coerce')>=i) & (pd.to_numeric(data["X"], errors='coerce')<i+n) &
                                     (pd.to_numeric(data["Y"], errors='coerce')>=j) & (pd.to_numeric(data["Y"], errors='coerce')<j+n)
                                     &(pd.to_numeric(data["Z"], errors='coerce')>=k) & (pd.to_numeric(data["Z"], errors='coerce')<k+m)]
                blocks.append(sub_block)
    blocks1 = []
    for i,j in enumerate(blocks):
        if len(j)>=5:
            blocks1.append(j)
    for i, j in enumerate(blocks1):
        blocks1[i]['blocks'] = i
    df2_new = pd.concat(blocks1)   
    
    
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5,shuffle = True, random_state = 2)
    blocks1_train = []
    blocks1_test = []
    for i in range(len(blocks1)):
        train = blocks1[i].iloc[list(kf.split(blocks1[i]))[1][0]]
        test = blocks1[i].iloc[list(kf.split(blocks1[i]))[1][1]]
        blocks1_train.append(train)
        blocks1_test.append(test)
    
    df2_train = pd.concat(blocks1_train)   
    df2_test = pd.concat(blocks1_test)           

    block_idxs1 = np.array(df2_train['blocks'])
    n_blocks = len(df2_train['blocks'].unique())
    
    Au_mean = []
    Au_var = []
    Cu_mean = []
    Cu_var = []
    
    for i in blocks1:
        Au_mean.append(np.mean(i['AU_ppm_log_noise']))
        Au_var.append(np.var(i['AU_ppm_log_noise']))
        Cu_mean.append(np.mean(i['CU_wt_log_noise']))
        Cu_var.append(np.var(i['CU_wt_log_noise']))
        
    
    plt.figure(figsize=(10, 6))    
    plt.errorbar(np.arange(0,50,1), Au_mean[0:50], yerr=Au_var[0:50], fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0);    
    plt.xlabel('Block number',fontsize=14)
    plt.ylabel('Log Au',fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    # plt.errorbar(np.arange(0,50,1), Cu_mean[0:50], yerr=Cu_var[0:50], fmt='o', color='black',
    #          ecolor='lightgray', elinewidth=3, capsize=0);    
    # plt.xlabel('Block number')
    # plt.ylabel('Log Cu')    
    # plt.ylim(-10,2)
    
    
        
    # plt.hist(df2_new.groupby(['blocks']).size(),bins=np.arange(0,130,1),color='b')
    # plt.xlabel('numbers of bore core data for each block', fontsize=14)
    # plt.ylabel('frequency', fontsize=14)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)

    
    import pymc3 as pm

    with pm.Model() as pooled_model:
        alpha = pm.Normal('alpha',mu = 0, sd = 1)
        beta1 = pm.Normal('beta1',mu = 0, sd = 1)
        beta2 = pm.Normal('beta2',mu = 0, sd = 1)
        eps = pm.Uniform('eps', lower=0, upper=1)
        Au_mean = alpha + beta1*df2_train['CU_wt_log_noise'].values + beta2*df2_train['CU_wt_log_noise'].values**2
        Au = pm.Normal('Au', mu = Au_mean, sd = eps, observed = df2_train['AU_ppm_log_noise'])
    with pooled_model:
        pooled_trace = pm.sample(2000)
    #pm.traceplot(pooled_trace)
    
    with pm.Model() as unpooled_model:
        #mu_alpha = pm.Normal('mu_alpha',mu=0,sd=100)
        #sigma_alpha = pm.HalfCauchy('sigma_alpha',2)
        
        #mu_beta = pm.Normal('mu_beta',mu=0,sd=100)
        #sigma_beta = pm.HalfCauchy('sigma_beta',2)
        
        #eps = pm.Uniform('eps',lower=0, upper=100)
        eps = pm.Uniform('eps', lower=0, upper=1)
        alpha = pm.Normal('alpha',mu = 0,sd = 1,shape = n_blocks)
        beta1 = pm.Normal('beta1',mu = 0,sd = 1, shape = n_blocks)
        beta2 = pm.Normal('beta2',mu = 0,sd = 1, shape = n_blocks)
        Au_mean = alpha[block_idxs1] + beta1[block_idxs1]*df2_train['CU_wt_log_noise'].values + beta2[block_idxs1]*df2_train['CU_wt_log_noise'].values**2
        Au = pm.Normal('Au', mu = Au_mean, sd = eps,observed = df2_train['AU_ppm_log_noise'])
        
    with unpooled_model:
        unpooled_trace = pm.sample(2000)
    # with pm.Model() as hirearchical_model_exponential:
    #     # lam_alpha = pm.Uniform('sigma_alpha', lower=0, upper=1)
    #     # lam_beta = pm.Uniform('sigma_beta', lower=0, upper=1)
        
    #     lam_alpha = pm.Normal('sigma_alpha',mu = 1, sd = 1)
    #     lam_beta = pm.Normal('sigma_beta',mu = 1, sd = 1)
    #     eps = pm.Uniform('eps', lower=0, upper=1)
        
    #     alpha = pm.Exponential('alpha',lam = lam_alpha, shape = n_blocks)
    #     beta = pm.Exponential('beta',lam = lam_beta,shape = n_blocks)
        
    #     Fe_mean = alpha[block_idxs1] + beta[block_idxs1]*df2_train['CU_wt_log_noise'].values
    #     Fe = pm.Normal('Fe', mu = Fe_mean, sd = eps,observed = df2_train['AU_ppm_log_noise'], shape = n_blocks) 
    # with hirearchical_model_exponential:
    #     hirearchical_trace_exponential = pm.sample(2000)    
        
    # with pm.Model() as hirearchical_model_lognormal:
    #     # lam_alpha = pm.Uniform('sigma_alpha', lower=0, upper=1)
    #     # lam_beta = pm.Uniform('sigma_beta', lower=0, upper=1)
        
    #     mu_alpha = pm.Normal('sigma_alpha',mu = 0, sd = 1)
    #     mu_beta = pm.Normal('sigma_beta',mu = 0, sd = 1)
    #     eps = pm.Uniform('eps', lower=0, upper=1)
        
    #     alpha = pm.LogNormal('alpha',mu = mu_alpha, shape = n_blocks)
    #     beta = pm.LogNormal('beta',mu = mu_beta,shape = n_blocks)
        
    #     Fe_mean = alpha[block_idxs1] + beta[block_idxs1]*df2_train['CU_wt_log_noise'].values
    #     Fe = pm.Normal('Fe', mu = Fe_mean, sd = eps,observed = df2_train['AU_ppm_log_noise'], shape = n_blocks) 
    # with hirearchical_model_lognormal:
    #     hirearchical_trace_lognormal = pm.sample(2000)        
    with pm.Model() as hirearchical_model_normal:
        mu_alpha = pm.Normal('mu_alpha',mu=0,sd=1)
        sigma_alpha = pm.Uniform('sigma_alpha', lower=0, upper=1)
        
        mu_beta1 = pm.Normal('mu_beta1',mu=0,sd=1)
        sigma_beta1 = pm.Uniform('sigma_beta1', lower=0, upper=1)
        
        mu_beta2 = pm.Normal('mu_beta2',mu=0,sd=1)
        sigma_beta2 = pm.Uniform('sigma_beta2', lower=0, upper=1)
        
        eps = pm.Uniform('eps', lower=0, upper=1)
        
        alpha = pm.Normal('alpha',mu = mu_alpha,sd = sigma_alpha, shape = n_blocks)
        beta1 = pm.Normal('beta1',mu = mu_beta1,sd = sigma_beta1, shape = n_blocks)
        beta2 = pm.Normal('beta2',mu = mu_beta2,sd = sigma_beta2, shape = n_blocks)
        
        Au_mean = alpha[block_idxs1] + beta1[block_idxs1]*df2_train['CU_wt_log_noise'].values + beta2[block_idxs1]*df2_train['CU_wt_log_noise'].values**2
        Au = pm.Normal('Au', mu = Au_mean, sd = eps,observed = df2_train['AU_ppm_log_noise'], shape = n_blocks)
        
    with hirearchical_model_normal:
        hirearchical_trace_normal = pm.sample(2000)    
        
    with pm.Model() as hirearchical_model_exponential:
        # lam_alpha = pm.Uniform('sigma_alpha', lower=0, upper=1)
        # lam_beta = pm.Uniform('sigma_beta', lower=0, upper=1)
        
        lam_alpha = pm.Normal('sigma_alpha',mu = 1, sd = 1)
        lam_beta1 = pm.Normal('sigma_beta1',mu = 1, sd = 1)
        lam_beta2 = pm.Normal('sigma_beta2',mu = 1, sd = 1)
        eps = pm.Uniform('eps', lower=0, upper=1)
        
        alpha = pm.Exponential('alpha',lam = lam_alpha, shape = n_blocks)
        beta1 = pm.Exponential('beta1',lam = lam_beta1,shape = n_blocks)
        beta2 = pm.Exponential('beta2',lam = lam_beta2,shape = n_blocks)
        
        
        Fe_mean = alpha[block_idxs1] + beta1[block_idxs1]*df2_train['CU_wt_log_noise'].values + beta2[block_idxs1]*df2_train['CU_wt_log_noise'].values**2
        Fe = pm.Normal('Fe', mu = Fe_mean, sd = eps,observed = df2_train['AU_ppm_log_noise'], shape = n_blocks) 
    with hirearchical_model_exponential:
        hirearchical_trace_exponential = pm.sample(2000)       
        
        
    pm.traceplot(hirearchical_trace_normal)
    #completepooling = az.summary(pooled_trace,round_to=5)
    #nopooling = az.summary(unpooled_trace,round_to=5)
    #partialpooling = az.summary(hirearchical_trace,round_to=5)
    
    # az.plot_trace(pooled_trace,figsize=(14,14))
    # az.plot_trace(unpooled_trace,figsize=(14,14))
    # az.plot_trace(hirearchical_trace,figsize=(14,14))

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
            
            
    hier_a = hirearchical_trace_normal['alpha'][:].mean(axis=0)[0:100]
    hier_b = hirearchical_trace_normal['beta'][:].mean(axis=0)[0:100]
    indv_a = unpooled_trace['alpha'][:].mean(axis=0)[0:100]
    indv_b = unpooled_trace['beta'][:].mean(axis=0)[0:100]
    compl_a = pooled_trace['alpha'][:].mean(axis=0)
    compl_b = pooled_trace['beta'][:].mean(axis=0)
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, xlabel='alpha', ylabel='beta',  
                         xlim=(-5, 5), ylim=(-2,3))
    ax.set_xlabel('alpha',fontsize=20)
    ax.set_ylabel('beta',fontsize=20)
    #ax.axhline(y=0,color='k',linewidth=3)
    for i in range(len(indv_b)):  
        ax.arrow(indv_a[i], indv_b[i], hier_a[i] - indv_a[i], hier_b[i] - indv_b[i], 
                 fc="k", ec="k", length_includes_head=True, alpha= 1 , head_width=.04)    
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.scatter(indv_a,indv_b,c='m', s=50, alpha=1, label = 'no pooling')
    ax.scatter(hier_a,hier_b, c='b', s=50, alpha=1, label = 'partial pooling')
    ax.scatter(compl_a,compl_b,c='r',marker='o', s=200, alpha=1, label = 'complete pooling')
    ax.legend(fontsize=20);
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
    # n=2 
    # seq1 = np.linspace(-1,0.5,6) 
    # n=60
    # seq1 = np.linspace(-4,0,6) 
    
    # n=10
    # seq1 = np.linspace(df2_train[df2_train['blocks']==n]['CU_wt_log_noise'].min(),df2_train[df2_train['blocks']==n]['CU_wt_log_noise'].max(),6) 
    # pred1 = np.zeros((len(seq1),len(pooled_trace[:][::10])*pooled_trace.nchains))
    # for i, w in enumerate(seq1):
    #     pred1[i] = pooled_trace[:][::10]['alpha'] + pooled_trace[:][::10]['beta']*w
    # seq2 = np.linspace(df2_train[df2_train['blocks']==n]['CU_wt_log_noise'].min(),df2_train[df2_train['blocks']==n]['CU_wt_log_noise'].max(),6) 
    # pred2 = np.zeros((len(seq2),len(unpooled_trace[:][::10])*unpooled_trace.nchains))
    # for i, w in enumerate(seq2):
    #     pred2[i] = unpooled_trace[:][::10]['alpha'][:,n] + unpooled_trace[:][::10]['beta'][:,n]*w
    # seq3 = np.linspace(df2_train[df2_train['blocks']==n]['CU_wt_log_noise'].min(),df2_train[df2_train['blocks']==n]['CU_wt_log_noise'].max(),6) 
    # pred3 = np.zeros((len(seq3),len(hirearchical_trace_normal[:][::10])*hirearchical_trace_normal.nchains))
    # for i, w in enumerate(seq3):
    #     pred3[i] = hirearchical_trace_normal[:][::10]['alpha'][:,n] + hirearchical_trace_normal[:][::10]['beta'][:,n]*w
    #plt.plot(seq1, pred1, '.',color='r')
    ##################error bar################
    # plt.figure(figsize=(10, 6))
    # x1 = np.linspace(df2_train[df2_train['blocks']==n]['CU_wt_log_noise'].min(),df2_train[df2_train['blocks']==n]['CU_wt_log_noise'].max(),6) 
    # y1 = pred1.mean(axis=1)
    # y_min1 = pred1.mean(axis=1) - pred1.min(axis=1)
    # y_max1 = pred1.max(axis=1) - pred1.mean(axis=1)
    # yerr = np.vstack((y_min1,y_max1))
    # plt.errorbar(x1,y1,yerr=yerr,fmt='o',color='c',alpha=0.5,label='complete pooling')

    # x2 = np.linspace(df2_train[df2_train['blocks']==n]['CU_wt_log_noise'].min(),df2_train[df2_train['blocks']==n]['CU_wt_log_noise'].max(),6) 
    # y2 = pred2.mean(axis=1)
    # y_min2 = pred2.mean(axis=1) - pred2.min(axis=1)
    # y_max2 = pred2.max(axis=1) - pred2.mean(axis=1)
    # yerr2 = np.vstack((y_min2,y_max2))
    # y_max2 + y_min2
    # plt.errorbar(x2,y2,yerr=yerr2,fmt='o',color='b',alpha=0.5,label='no pooling')
    
    # x3 = np.linspace(df2_train[df2_train['blocks']==n]['CU_wt_log_noise'].min(),df2_train[df2_train['blocks']==n]['CU_wt_log_noise'].max(),6) 
    # y3 = pred3.mean(axis=1)
    # y_min3 = pred3.mean(axis=1) - pred3.min(axis=1)
    # y_max3 = pred3.max(axis=1) - pred3.mean(axis=1)
    # y_max3 + y_min3
    # yerr3 = np.vstack((y_min3,y_max3))
    # plt.errorbar(x3,y3,yerr=yerr3,fmt='o',color='r',alpha=0.5,label='partial pooling (hierarchical)')
    # plt.scatter(df2_train[df2_train['blocks']==n]['CU_wt_log_noise'], df2_train[df2_train['blocks']==n]['AU_ppm_log_noise'],label='bore core data',s=200,color='k',marker = '.')
    # #plt.legend(fontsize=14,loc='lower left')
    # plt.xlabel('log Cu (wt)',fontsize=14)
    # plt.ylabel('log Au (ppm)',fontsize=14)
    # #plt.ylim(-5,10)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.title('Blocks No.' + str(df2_train[df2_train['blocks']==n]['blocks'].unique()[0]+1))
    # plt.show()







    #plt.scatter(df2_train[df2_train['elevation_inxs']==1]['CuT_dh_log_noise'], df2_train[df2_train['elevation_inxs']==1]['Fe_dh_log_noise'],label='bore core')
    #plt.plot(seq1, pred1.mean(1), 'k')
    # az.plot_hdi(seq1, pred1.T,color='c')
    # az.plot_hdi(seq2, pred2.T,color='r')
    # az.plot_hdi(seq3, pred3.T,color='g')
    # plt.legend()
    
    # seq = np.linspace(-5,1,20) 
    # pred = np.zeros((len(seq),len(unpooled_trace[:][::10])*unpooled_trace.nchains))
    # for i, w in enumerate(seq):
    #     pred[i] = unpooled_trace[:][::10]['alpha'][:,1] + unpooled_trace[:][::10]['beta'][:,1]*w
    # plt.scatter(df2_train[df2_train['blocks']==1]['CU_wt_log_noise'], df2_train[df2_train['blocks']==1]['AU_ppm_log_noise'])
    # plt.plot(seq, pred.mean(1), 'k')
    # #az.plot_hdi(seq, pred.T)

    # seq1 = np.linspace(-5,1,20) 
    # pred1= np.zeros((len(seq1),len(hirearchical_trace_normal[:][::10])*hirearchical_trace_normal.nchains))
    # for i, w in enumerate(seq1):
    #     pred1[i] = hirearchical_trace_normal[:][::10]['alpha'][:,1] + hirearchical_trace_normal[:][::10]['beta'][:,1]*w
    # plt.scatter(df2_train[df2_train['blocks']==1]['CU_wt_log_noise'], df2_train[df2_train['blocks']==1]['AU_ppm_log_noise'])
    # plt.plot(seq1, pred1.mean(1), 'k')
    # #az.plot_hdi(seq, pred.T)
    # #az.plot_hdi(seq1, pred1.T)
    
    # df_test_label = [val for val in list(df2_train.blocks.unique()) for _ in (0, 1)]

    # df_label = [val for val in list(df2_train.blocks.unique()) for _ in (0, 1)]



    #selection2 = [3,6,20,33,49,64]  #list(df2_new.blocks.unique())[493:499] #0-3 #246-249 #493-496
    selection2 = [1,2,3,4,5,6]
    fig,axis = plt.subplots(2,3,figsize=(20,8),sharey=True,sharex=False);
    axis = axis.ravel()
    for i,c in enumerate(selection2):
        c_data = df2_test.loc[df2_test.blocks==c]
        c_data = c_data.reset_index(drop=True)
        xvals = np.linspace(c_data['CU_wt_log_noise'].min()-1,c_data['CU_wt_log_noise'].max()+1)
        
        axis[i].set_title('Block No.' + str(c+1),fontsize=14)
        axis[i].scatter(c_data['CU_wt_log_noise'],c_data['AU_ppm_log_noise'],color='k',marker='.',s=200,label = 'bore core')
        axis[i].plot(xvals,hirearchical_trace_exponential['alpha'][:,c].mean()+hirearchical_trace_exponential['beta1'][:,c].mean()*xvals + hirearchical_trace_exponential['beta2'][:,c].mean()*xvals**2,'c',alpha=1,lw=3.,label='partial pooled exponential')
        axis[i].plot(xvals,hirearchical_trace_normal['alpha'][:,c].mean()+hirearchical_trace_normal['beta1'][:,c].mean()*xvals+hirearchical_trace_normal['beta2'][:,c].mean()*xvals**2,'m',alpha=1,lw=3.,label='partial pooled normal')
        #axis[i].plot(xvals,hirearchical_trace_lognormal['alpha'][:,c].mean()+hirearchical_trace_lognormal['beta'][:,c].mean()*xvals,'black',alpha=1,lw=3.,label='partial pooled lognormal')
        axis[i].plot(xvals,unpooled_trace['alpha'][:,c].mean()+unpooled_trace['beta1'][:,c].mean()*xvals+unpooled_trace['beta2'][:,c].mean()*xvals**2 ,'r',alpha=1,lw=3.,label='no pooled')
        axis[i].plot(xvals,pooled_trace['alpha'][:].mean()+pooled_trace['beta1'][:].mean()*xvals+pooled_trace['beta2'][:].mean()*xvals**2,'g',alpha=1,lw=3.,label='complete pooled')
        # axis[i].set_xlim(-5,3)
        # axis[i].set_ylim(-5,3)
        axis[i].tick_params(axis='both', which='major', labelsize=14)
        axis[i].set_ylabel('log Au (ppm)',fontsize=14)
        if i>2:
            axis[i].set_xlabel('log Cu (wt)',fontsize=14)
        if i==5:
            axis[i].legend(loc='lower right',prop={'size':12})
            
            
            
    from sklearn.metrics import mean_squared_error

    mse_no_pooling = []
    for i in range(len(blocks1_test)):
        estimate = np.mean(unpooled_trace['beta2'][:],axis=0)[i]*blocks1_test[i]['CU_wt_log_noise']**2 + blocks1_test[i]['CU_wt_log_noise']*np.mean(unpooled_trace['beta1'][:],axis=0)[i] + np.mean(unpooled_trace['alpha'][:],axis=0)[i] 
        mse_no_pooling.append(mean_squared_error(blocks1_test[i]['AU_ppm_log_noise'],estimate))    
    print(np.mean(mse_no_pooling))
    
    mse_partial_pooling_normal = []
    for i in range(len(blocks1_test)):
        estimate = np.mean(hirearchical_trace_normal['beta2'][:],axis=0)[i]*blocks1_test[i]['CU_wt_log_noise']**2 + blocks1_test[i]['CU_wt_log_noise']*np.mean(hirearchical_trace_normal['beta1'][:],axis=0)[i] + np.mean(hirearchical_trace_normal['alpha'][:],axis=0)[i] 
        mse_partial_pooling_normal.append(mean_squared_error(blocks1_test[i]['AU_ppm_log_noise'],estimate))    
    print(np.mean(mse_partial_pooling_normal))

    mse_partial_pooling_exponential = []
    for i in range(len(blocks1_test)):
        estimate = np.mean(hirearchical_trace_exponential['beta2'][:],axis=0)[i]*blocks1_test[i]['CU_wt_log_noise']**2 + blocks1_test[i]['CU_wt_log_noise']*np.mean(hirearchical_trace_exponential['beta1'][:],axis=0)[i] + np.mean(hirearchical_trace_normal['alpha'][:],axis=0)[i] 
        mse_partial_pooling_exponential.append(mean_squared_error(blocks1_test[i]['AU_ppm_log_noise'],estimate))    
    print(np.mean(mse_partial_pooling_exponential))
    # mse1 = mse[mse['exponential-normal']<0.2]
    
    # np.mean(mse1['no pooled'])
    # np.mean(mse1['partial pooled normal'])
    # np.mean(mse1['partial pooled exponential'])
    
    # plt.scatter(np.arange(0,50,1), mse_no_pooling[0:50], color='black')
    # plt.scatter(np.arange(0,50,1), mse_partial_pooling_normal[0:50], color='red')
    # plt.scatter(np.arange(0,50,1), mse_partial_pooling_exponential[0:50], color='c')
    # plt.xlabel('Block number',fontsize=14)
    # plt.ylabel('MSE',fontsize=14)
    # plt.ylim(0,1)
    # plt.tick_params(axis='both', which='major', labelsize=14)








    partialpooling_normal_alpha_mean = np.mean(hirearchical_trace_normal['alpha'][:,0:50],axis=0)
    partialpooling_normal_alpha_std = np.std(hirearchical_trace_normal['alpha'][:,0:50],axis=0)
    
    nopooling_alpha_mean = np.mean(unpooled_trace['alpha'][:,0:50],axis=0)
    nopooling_alpha_std = np.std(unpooled_trace['alpha'][:,0:50],axis=0)
                
    partialpooling_normal_beta_mean = np.mean(hirearchical_trace_normal['beta2'][:,0:50],axis=0)
    partialpooling_normal_beta_std = np.std(hirearchical_trace_normal['beta2'][:,0:50],axis=0)

    nopooling_beta_mean = np.mean(unpooled_trace['beta2'][:,0:50],axis=0)
    nopooling_beta_std = np.std(unpooled_trace['beta2'][:,0:50],axis=0)
    

    # plt.errorbar(np.arange(0,50,1), partialpooling_alpha_mean, partialpooling_alpha_std, fmt='o', color='black',
    #           ecolor='lightgray', elinewidth=3, capsize=0);    
    # plt.xlabel('Block number',fontsize=14)
    # plt.ylabel('Alpha',fontsize=14)
    # plt.tick_params(axis='both', which='major', labelsize=14)
    
    # plt.figure(figsize=(10, 6))    
    # plt.errorbar(np.arange(0,50,1), nopooling_alpha_mean, nopooling_alpha_std, fmt='o', color='black',
    #           ecolor='lightgray', elinewidth=3, capsize=0);    
    # plt.errorbar(np.arange(0,50,1), partialpooling_alpha_mean, partialpooling_alpha_std, fmt='o', color='red',
    #           ecolor='lightgreen', elinewidth=3, capsize=0);    
    plt.figure(figsize=(10, 6))    
    plt.scatter(np.arange(0,50,1), nopooling_beta_mean, color='black')
    plt.scatter(np.arange(0,50,1), partialpooling_normal_beta_mean, color='red')
    plt.scatter(np.arange(0,50,1), partialpooling_exponential_beta_mean, color='c')
    plt.axhline(y=compl_b, color='g', linestyle='-')
    plt.xlabel('Block number',fontsize=14)
    plt.ylabel('Alpha',fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
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

    # df_train_label = [val for val in list(df2_new.blocks.unique()) for _ in (0, 1)]
    # df_train_label1 = [val for val in df_train_label for _ in (0, 1)]
    # df_train_label1 = pd.DataFrame(df_train_label1,columns=['blocks'])
    # df_train_label1_drop = df_train_label1.drop_duplicates(keep='first')
    # df_train_label2 = df_train_label1.drop(df_train_label1_drop.index)['blocks'].to_list()
    
    ###################plot complete/no/partial pooling regression on training/testing data####################
    selection2 = [1,1,1,2,2,2,3,3,3]
    fig,axis = plt.subplots(3,3,figsize=(20,12),sharey=True,sharex=True);
    axis = axis.ravel()
    for i,c in enumerate(selection2):
        c_data = df2_train.loc[df2_train.blocks==c]
        c_data = c_data.reset_index(drop=True)
        xvals = np.linspace(-5,3)
        if (i-1)%3 ==1:#258
            axis[i].scatter(c_data['CU_wt_log_noise'],c_data['AU_ppm_log_noise'],color='k',marker='.',s=200,label = 'bore core')
            axis[i].set_title('Block No.' + str(c+1),fontsize=14)
            num=1
            for a_val, b_val1, b_val2 in zip(hirearchical_trace_exponential['alpha'][:,c],hirearchical_trace_exponential['beta1'][:,c],hirearchical_trace_exponential['beta2'][:,c]):
                if num==1:
                    axis[i].plot(xvals,a_val+b_val1*xvals + b_val2*xvals**2,'c',label = 'partial pooled exponential')
                    num+=1
                else:
                    axis[i].plot(xvals,a_val+b_val1*xvals + b_val2*xvals**2,'c',alpha=.01)

        elif (i+1)%3 ==2: #147
            axis[i].scatter(c_data['CU_wt_log_noise'],c_data['AU_ppm_log_noise'],color='k',marker='.',s=200,label = 'bore core')
            axis[i].set_title('Block No.' + str(c+1),fontsize=14)
            num=1
            for a_val, b_val1, b_val2 in zip(hirearchical_trace_normal['alpha'][:,c],hirearchical_trace_normal['beta1'][:,c],hirearchical_trace_normal['beta2'][:,c]):
                if num==1:
                    axis[i].plot(xvals,a_val+b_val1*xvals + b_val2*xvals**2,'c',label = 'partial pooled normal')
                    num+=1
                else:
                    axis[i].plot(xvals,a_val+b_val1*xvals + b_val2*xvals**2,'c',alpha=.01)
                    
        else:
            axis[i].scatter(c_data['CU_wt_log_noise'],c_data['AU_ppm_log_noise'],color='k',marker='.',s=200,label = 'bore core')  
            axis[i].set_title('Block No.' + str(c+1),fontsize=14)
            num=1
        #axis[i].plot(xvals,hirearchical_trace['alpha'][1000:,c_index].mean()+hirearchical_trace['beta'][1000:,c_index].mean()*xvals,'r',alpha=1,lw=1.,label='hirearchical')
            for a_val, b_val1, b_val2 in zip(unpooled_trace['alpha'][:,c],unpooled_trace['beta1'][:,c],unpooled_trace['beta2'][:,c]):  
                if num==1:
                    axis[i].plot(xvals,a_val+b_val1*xvals + b_val2*xvals**2,'r',label = 'no pooled')
                    num+=1
                else:
                    axis[i].plot(xvals,a_val+b_val1*xvals+b_val2*xvals**2,'r',alpha=.01)

        #axis[i].plot(xvals,unpooled_trace['alpha'][1000:,c_index].mean()+unpooled_trace['beta'][1000:,c_index].mean()*xvals,'g',alpha=1,lw=1.,label='individual')
        # axis[i].set_xlim(-5,3)
        # axis[i].set_ylim(-5,3)
        axis[i].legend(loc='lower right',fontsize=14)
        axis[i].set_xlabel('log Cu',fontsize=14)
        axis[i].set_ylabel('log Fe',fontsize=14)
        axis[i].tick_params(axis='both', which='major', labelsize=14)
    
    
    
    
    
    
    # import warnings
    # import matplotlib.pyplot as plt
    # import plotly.express as px
    # import numpy as np
    # import pandas as pd    
    # import plotly.io as pio
    # import plotly.graph_objs as go
    # import arviz as az
    # from scipy import linalg, stats
    # import time
    
    # def gelman_rubin(data):
    #     """
    #     Apply Gelman-Rubin convergence diagnostic to a bunch of chains.
    #     :param data: np.array of shape (Nchains, Nsamples, Npars)
    #     """
    #     Nchains, Nsamples, Npars = data.shape
    #     B_on_n = data.mean(axis=1).var(axis=0)      # variance of in-chain means
    #     W = data.var(axis=1).mean(axis=0)           # mean of in-chain variances

    #     # simple version, as in Obsidian -- not reliable on its own!
    #     sig2 = (Nsamples/(Nsamples-1))*W + B_on_n
    #     Vhat = sig2 + B_on_n/Nchains
    #     Rhat = Vhat/W

    #     # advanced version that accounts for ndof
    #     m, n = np.float(Nchains), np.float(Nsamples)
    #     si2 = data.var(axis=1)
    #     xi_bar = data.mean(axis=1)
    #     xi2_bar = data.mean(axis=1)**2
    #     var_si2 = data.var(axis=1).var(axis=0)
    #     allmean = data.mean(axis=1).mean(axis=0)
    #     cov_term1 = np.array([np.cov(si2[:,i], xi2_bar[:,i])[0,1]
    #                           for i in range(Npars)])
    #     cov_term2 = np.array([-2*allmean[i]*(np.cov(si2[:,i], xi_bar[:,i])[0,1])
    #                           for i in range(Npars)])
    #     var_Vhat = ( ((n-1)/n)**2 * 1.0/m * var_si2
    #              +   ((m+1)/m)**2 * 2.0/(m-1) * B_on_n**2
    #              +   2.0*(m+1)*(n-1)/(m*n**2)
    #                     * n/m * (cov_term1 + cov_term2))
    #     df = 2*Vhat**2 / var_Vhat
    #     print ("gelman_rubin(): var_Vhat = {}, df = {}".format(var_Vhat, df))
    #     Rhat *= df/(df-2)
        
    #     return Rhat
    # def autocorr(x, D, plot=True):
    #     """
    #     Discrete autocorrelation function + integrated autocorrelation time.
    #     Calculates directly, though could be sped up using Fourier transforms.
    #     See Daniel Foreman-Mackey's tutorial (based on notes from Alan Sokal):
    #     https://emcee.readthedocs.io/en/stable/tutorials/autocorr/

    #     :param x: np.array of data, of shape (Nsamples, Ndim)
    #     :param D: number of return arrays
    #     """
    #     # Baseline discrete autocorrelation:  whiten the data and calculate
    #     # the mean sample correlation in each window
    #     xp = np.atleast_2d(x)
    #     z = (xp-np.mean(xp, axis=0))/np.std(xp, axis=0)
    #     Ct = np.ones((D, z.shape[1]))
    #     Ct[1:,:] = np.array([np.mean(z[i:]*z[:-i], axis=0) for i in range(1,D)])
    #     # Integrated autocorrelation tau_hat as a function of cutoff window M
    #     tau_hat = 1 + 2*np.cumsum(Ct, axis=0)
    #     # Sokal's advice is to take the autocorrelation time calculated using
    #     # the smallest integration limit M that's less than 5*tau_hat[M]
    #     Mrange = np.arange(len(tau_hat))
    #     tau = np.argmin(Mrange[:,None] - 5*tau_hat, axis=0)
    #     print("tau =", tau)
    #     # Plot if requested
    #     if plot:
    #         fig = plt.figure(figsize=(6,4))
    #         plt.plot(Ct)
    #         plt.title('Discrete Autocorrelation ($\\tau = {:.1f}$)'.format(np.mean(tau)))
    #     return np.array(Ct), tau
    # def traceplots(x, xnames=None, title=None):
    #     """
    #     Runs trace plots.
    #     :param x:  np.array of shape (N, d)
    #     :param xnames:  optional iterable of length d, containing the names
    #         of variables making up the dimensions of x (used as y-axis labels)
    #     :param title:  optional plot title
    #     """
    #     # set out limits of plot spaces, in dimensionless viewport coordinates
    #     # that run from 0 (bottom, left) to 1 (top, right) along both axes
    #     N, d = x.shape
    #     fig = plt.figure()
    #     left, tracewidth, histwidth = 0.1, 0.65, 0.15
    #     bottom, rowheight = 0.1, 0.8/d
    #     spacing = 0.05
        
    #     for i in range(d):
    #         # Set the location of the trace and histogram viewports,
    #         # starting with the first dimension from the bottom of the canvas
    #         rowbottom = bottom + i*rowheight
    #         rect_trace = (left, rowbottom, tracewidth, rowheight)
    #         rect_hist = (left + tracewidth, rowbottom, histwidth, rowheight)
    #         # First set of trace plot axes
    #         if i == 0:
    #             ax_trace = fig.add_axes(rect_trace)
    #             ax_trace.plot(x[:,i])
    #             ax_trace.set_xlabel("Sample Count")
    #             ax_tr0 = ax_trace
    #         # Other sets of trace plot axes that share the first trace's x-axis
    #         # Make tick labels invisible so they don't clutter up the plot
    #         elif i > 0:
    #             ax_trace = fig.add_axes(rect_trace, sharex=ax_tr0)
    #             ax_trace.plot(x[:,i])
    #             plt.setp(ax_trace.get_xticklabels(), visible=False)
    #         # Title at the top
    #         if i == d-1 and title is not None:
    #             plt.title(title)
    #         # Trace y-axis labels
    #         if xnames is not None:
    #             ax_trace.set_ylabel(xnames[i])
    #         # Trace histograms at the right
    #         ax_hist = fig.add_axes(rect_hist, sharey=ax_trace)
    #         ax_hist.hist(x[:,i], orientation='horizontal', bins=50)
    #         plt.setp(ax_hist.get_xticklabels(), visible=False)
    #         plt.setp(ax_hist.get_yticklabels(), visible=False)
    #         xlim = ax_hist.get_xlim()
    #         ax_hist.set_xlim([xlim[0], 1.1*xlim[1]])
    # def profile_timer(f, *args, **kwargs):
    #     """
    #     Times a function call f() and prints how long it took in seconds
    #     (to the nearest millisecond).
    #     :param func:  the function f to call
    #     :return:  same return values as f
    #     """
    #     t0 = time.time()
    #     result = f(*args, **kwargs)
    #     t1 = time.time()
    #     print ("time to run {}: {:.3f} sec".format(f.__name__, t1-t0))
    #     return result
    # class OutlierRegressionMixture(object):
        
    #     def __init__(self, y, phi_x, sigma2, V, p):
    #         self.y = y
    #         self.phi_x = phi_x
    #         self.sigma2 = sigma2
    #         self.V = V
    #         self.p = p
            
    #     def log_likelihood(self, theta):
    #         """
    #         Mixture likelihood accounting for outliers
    #         """
    #         # Form regression mean and residuals
    #         w = theta
    #         resids = self.y - np.dot(w, self.phi_x)
    #         # Each mixture component is a Gaussian with baseline or inflated variance
    #         S2_in, S2_out = self.sigma2, self.sigma2 + self.V
    #         exp_in  = np.exp(-0.5*resids**2/S2_in)/np.sqrt(2*np.pi*S2_in)
    #         exp_out = np.exp(-0.5*resids**2/S2_out)/np.sqrt(2*np.pi*S2_out)
    #         # The final log likelihood sums over the log likelihoods for each point
    #         logL = np.sum(np.log((1-self.p)*exp_in + self.p*exp_out))
    #         return logL

    #     def log_prior(self, theta):
    #         """
    #         Priors over parameters 
    #         """
    #         # DANGER:  improper uniform for now, assume data are good enough
    #         return 0.0
            
    #     def log_posterior(self, theta):
    #         logpost = self.log_prior(theta) + self.log_likelihood(theta)
    #         if np.isnan(logpost):
    #             return -np.inf
    #         return logpost
        
    #     def __call__(self, theta):
    #         return self.log_posterior(theta)
    # class GaussianProposal(object):
    #     """
    #     A standard isotropic Gaussian proposal for Metropolis Random Walk.
    #     """
        
    #     def __init__(self, stepsize):
    #         """
    #         :param stepsize:  either float or np.array of shape (d,)
    #         """
    #         self.stepsize = stepsize
            
    #     def __call__(self, theta):
    #         """
    #         :param theta:  parameter vector = np.array of shape (d,)
    #         :return: tuple (logpost, logqratio)
    #             logpost = log (posterior) density p(y) for the proposed theta
    #             logqratio = log(q(x,y)/q(y,x)) for asymmetric proposals
    #         """
    #         # this proposal is symmetric so the Metropolis q-ratio is 1
    #         return theta + self.stepsize*np.random.normal(size=theta.shape), 0.0
    # class MHSampler(object):
    #     """
    #     Run a Metropolis-Hastings algorithm given a Model and Proposal.
    #     """

    #     def __init__(self, model, proposal, debug=False):
    #         """
    #         Initialize a Sampler with a model, a proposal, data, and a guess
    #         at some reasonable starting parameters.
    #         :param model: callable accepting a np.array parameter vector
    #             of shape matching the initial guess theta0, and returning
    #             a probability (such as a posterior probability)
    #         :param proposal: callable accepting a np.array parameter vector
    #             of shape matching the initial guess theta0, and returning
    #             a proposal of the same shape, as well as the log ratio
    #                 log (q(theta'|theta)/q(theta|theta'))
    #         :param theta0: np.array of shape (Npars,)
    #         :param debug: Boolean flag for whether to turn on the debugging
    #             print messages in the sample() method
    #         """
    #         self.model = model
    #         self.proposal = proposal
    #         self._chain_thetas = [ ]
    #         self._chain_logPs = [ ]
    #         self._debug = debug

    #     def run(self, theta0, Nsamples):
    #         """
    #         Run the Sampler for Nsamples samples.
    #         """
    #         self._chain_thetas = [ theta0 ]
    #         self._chain_logPs = [ self.model(theta0) ]
    #         for i in range(Nsamples):
    #             theta, logpost = self.sample()
    #             self._chain_thetas.append(theta)
    #             self._chain_logPs.append(logpost)
    #         self._chain_thetas = np.array(self._chain_thetas)
    #         self._chain_logPs = np.array(self._chain_logPs)

    #     def sample(self):
    #         """
    #         Draw a single sample from the MCMC chain, and accept or reject
    #         using the Metropolis-Hastings criterion.
    #         """
    #         theta_old = self._chain_thetas[-1]
    #         logpost_old = self._chain_logPs[-1]
    #         theta_prop, logqratio = self.proposal(theta_old)
    #         if logqratio is -np.inf:
    #             # flag that this is a Gibbs sampler, auto-accept and skip the rest,
    #             # assuming the modeler knows what they're doing
    #             return theta_prop, logpost
    #         logpost = self.model(theta_prop)
    #         mhratio = min(1, np.exp(logpost - logpost_old - logqratio))
    #         if self._debug:
    #             # this can be useful for sanity checks
    #             print("theta_old, theta_prop =", theta_old, theta_prop)
    #             print("logpost_old, logpost_prop =", logpost_old, logpost)
    #             print("logqratio =", logqratio)
    #             print("mhratio =", mhratio)
    #         if np.random.uniform() < mhratio:
    #             return theta_prop, logpost
    #         else:
    #             return theta_old, logpost_old
            
    #     def chain(self):
    #         """
    #         Return a reference to the chain.
    #         """
    #         return self._chain_thetas
        
    #     def accept_frac(self):
    #         """
    #         Calculate and return the acceptance fraction.  Works by checking which
    #         parameter vectors are the same as their predecessors.
    #         """
    #         samesame = (self._chain_thetas[1:] == self._chain_thetas[:-1])
    #         if len(samesame.shape) == 1:
    #             samesame = samesame.reshape(-1, 1)
    #         samesame = np.all(samesame, axis=1)
    #         return 1.0 - (np.sum(samesame) / np.float(len(samesame)))

    # class OutlierGibbsProposal(object):
    #     """
    #     A Gibbs sampling proposal to sample from this model.
    #     """
        
    #     def __init__(self, model):
    #         # Add access to the model just in case we need it
    #         self.y = model.y
    #         self.phi_x = model.phi_x
    #         self.sigma2 = model.sigma2
    #         self.V = model.V
    #         self.p = model.p
    #         self.Nw, self.Nq = model.phi_x.shape
    #         # Some pre-computed constants
    #         self.logP_q0_norm = -0.5*np.log(2*np.pi*self.sigma2) + np.log(1-self.p)
    #         self.logP_q1_norm = -0.5*np.log(2*np.pi*(self.sigma2 + self.V)) + np.log(self.p)
            
    #     def __call__(self, theta):
    #         """
    #         :param theta:  parameter vector = np.array of shape (d,)
    #         :return: tuple (logpost, logqratio)
    #             logpost = log (posterior) density p(y) for the proposed theta
    #             logqratio = log(q(x,y)/q(y,x)) for asymmetric proposals
    #         """
    #         w, q = theta[:self.Nw], theta[self.Nw:]

    #         # Step 1:  propose from P(q|w)
    #         # Each mixture component has a Gaussian variance which may be inflated
    #         # The final log likelihood sums over the log likelihoods for each point
    #         # Conditioned on the w's, figure out the density ratios for q
    #         resids = self.y - np.dot(w, self.phi_x)
    #         logP_q0 = -0.5*(resids**2/self.sigma2) + self.logP_q0_norm
    #         logP_q1 = -0.5*(resids**2/(self.sigma2 + self.V)) + self.logP_q1_norm
    #         logP_q = np.exp(logP_q1 - logP_q0)
    #         # Now propose a random set of q's based on those probabilities
    #         q_prop = (np.random.uniform(q.shape) < logP_q)
            
    #         # Step 2:  propose from P(w|q)
    #         # Conditioned on the q's, draw from the conditional density for the w's
    #         # Helpful that the covariance matrix is diagonal, thus easily inverted
    #         yvar = self.sigma2 + q_prop*self.V
    #         Cinv = np.diag(1.0/yvar)
    #         wprec = np.dot(self.phi_x, np.dot(Cinv, self.phi_x.T))
    #         Lprec = linalg.cholesky(wprec)
    #         # The conditional posterior for the weights is multi-Gaussian with
    #         # mu = (phi*C.inv*phi.T).inv * phi*C.inv*y, var = (phi*C.inv*phi.T).inv
    #         # More numerically stable to use linalg.solve than linalg.inv
    #         u = np.random.normal(size=w.shape)
    #         w_prop = linalg.solve_triangular(Lprec, u)
    #         w_prop += linalg.solve(wprec, np.dot(self.phi_x, self.y/yvar))

    #         # Step 3:  Profit!
    #         # This is a Gibbs step so I'm just going to return -np.inf for the
    #         # log proposal density ratio, which should make MHSampler auto-accept
    #         return np.concatenate([w_prop, q_prop]), -np.inf
     
    # class OutlierRegressionMixture(object):
        
    #     def __init__(self, y, phi_x, sigma2, V, p):
    #         self.y = y
    #         self.phi_x = phi_x
    #         self.sigma2 = sigma2
    #         self.V = V
    #         self.p = p
            
    #     def log_likelihood(self, theta):
    #         """
    #         Mixture likelihood accounting for outliers
    #         """
    #         # Form regression mean and residuals
    #         w = theta
    #         resids = self.y - np.dot(w, self.phi_x)
    #         # Each mixture component is a Gaussian with baseline or inflated variance
    #         S2_in, S2_out = self.sigma2, self.sigma2 + self.V
    #         exp_in  = np.exp(-0.5*resids**2/S2_in)/np.sqrt(2*np.pi*S2_in)
    #         exp_out = np.exp(-0.5*resids**2/S2_out)/np.sqrt(2*np.pi*S2_out)
    #         # The final log likelihood sums over the log likelihoods for each point
    #         logL = np.sum(np.log((1-self.p)*exp_in + self.p*exp_out))
    #         return logL

    #     def log_prior(self, theta):
    #         """
    #         Priors over parameters 
    #         """
    #         # DANGER:  improper uniform for now, assume data are good enough
    #         return 0.0
            
    #     def log_posterior(self, theta):
    #         logpost = self.log_prior(theta) + self.log_likelihood(theta)
    #         if np.isnan(logpost):
    #             return -np.inf
    #         return logpost
        
    #     def __call__(self, theta):
    #         return self.log_posterior(theta)    
    # # Stub for MCMC stuff

    # class OutlierRegressionLatent(object):
        
    #     def __init__(self, y, phi_x, sigma2, V, p):
    #         self.y = y
    #         self.phi_x = phi_x
    #         self.sigma2 = sigma2
    #         self.V = V
    #         self.p = p
            
    #     def log_likelihood(self, theta):
    #         """
    #         Mixture likelihood accounting for outliers
    #         """
    #         # Form regression mean and residuals
    #         w, q = theta
    #         resids = self.y - np.dot(w, self.phi_x)
    #         # Each mixture component has a Gaussian variance which may be inflated
    #         # The final log likelihood sums over the log likelihoods for each point
    #         S2 = self.sigma2 + q*self.V
    #         logL = -0.5*np.sum(resids**2/S2 + log(2*np.pi*S2))
    #         return logL

    #     def log_prior(self, theta):
    #         """
    #         Priors over parameters
    #         """
    #         # Bernoulli prior for the latents; leave improper uniforms over the weights
    #         # (don't do this at home, folks, we're just in a rush today)
    #         w, q = theta
    #         N, Nout = len(q), np.sum(q)
    #         return Nout*np.log(p) + (N-Nout)*np.log(1-p)
            
    #     def log_posterior(self, theta):
    #         logpost = self.log_prior(theta) + self.log_likelihood(theta)
    #         if np.isnan(logpost):
    #             return -np.inf
    #         return logpost
        
    #     def conditional_draw(self, theta, i):
    #         """
    #         A stub for Gibbs sampling
    #         """
    #         pass

    #     def __call__(self, theta):
    #         return self.log_posterior(theta)
    
    #df3= df2[300:600]#.sort_values(by=['CuT_dh'])[0:300]
    
    # X = np.array(df2_new[df2_new['blocks']==24].sort_values(by=['CU_wt_log'])['CU_wt_log'])
    # Y = np.array(df2_new[df2_new['blocks']==24].sort_values(by=['CU_wt_log'])['AU_ppm_log'])
    # sigma2 = 1
    # phi_x = np.vstack([X**0, X**1])
    # p=0.2
    # V=100.0
    # Nsamp = 10000
    # logpost_outl = OutlierRegressionLatent(Y, phi_x, sigma2, V, p)
    # sampler = MHSampler(lambda theta: np.inf, OutlierGibbsProposal(logpost_outl))
    # chain_array = [ ]
    # for i in range(5):
    #     theta0 = np.random.uniform(size=np.sum(phi_x.shape))
    #     profile_timer(sampler.run, np.array(theta0), Nsamp)
    #     print("chain.mean, chain.std =", sampler.chain().mean(), sampler.chain().std())
    #     print("acceptance fraction =", sampler.accept_frac())
    #     chain_array.append(sampler.chain())
    # chain_array = np.array(chain_array)
    # flatchain = chain_array.reshape(-1, chain_array.shape[-1])
    # traceplots(chain_array[1], #  xnames=['w0', 'w1', 'w2'],
    #            title="Outlier Regression Weight Traces")
    # rho_k, tau = autocorr(chain_array[1], 1000, plot=False)
    # print("chain_array.shape =", chain_array.shape)
    # print("chain.mean =", flatchain.mean(axis=0))
    # print("chain.std =", flatchain.std(axis=0))
    # print("tau.shape =", tau.shape)
    # Rhat = gelman_rubin(chain_array)
    # print("psrf =", Rhat)
    # wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))
    # # Visualize our answer!
    # plt.figure(figsize=(6,4))
    # plt.plot(X, Y, ls='None', marker='o', ms=3, label="Data")
    # plt.plot(X, np.dot(wML, phi_x), ls='--', lw=2, label="Maximum Likelihood")
    # flatchain = chain_array.reshape(-1, chain_array.shape[-1])
    # func_samples = np.dot(flatchain[:,:2], phi_x)
    # post_mu = np.mean(func_samples, axis=0)
    # post_sig = np.std(func_samples, axis=0)
    # plt.plot(X, post_mu, ls='--', lw=2, color='dodgerblue', label="Posterior Mean")
    # plt.fill_between(X, post_mu-post_sig, post_mu+post_sig, color='dodgerblue', alpha=0.5, label="Posterior Variance")
    # plt.legend(loc='best')
    # plt.xlabel("x")
    # plt.ylabel("y")
    # # plt.xlim(-5,1)
    # # plt.ylim(-5,3)
    # plt.title("Regression with Outliers (Latent)")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
