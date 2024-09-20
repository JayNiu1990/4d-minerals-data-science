if __name__ == '__main__':
    import warnings
    import matplotlib.pyplot as plt
    import plotly.express as px
    import numpy as np
    import pandas as pd    
    import plotly.io as pio
    import plotly.graph_objs as go
    import arviz as az
    from scipy import stats
    fields = ['BHID','Fe_dh','As_dh','CuT_dh',"X","Y","Z","LITH"]
    pio.renderers.default='browser'
    df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)
    df = df.dropna()
    df = df[(pd.to_numeric(df["CuT_dh"], errors='coerce')>0) & (pd.to_numeric(df["Fe_dh"], errors='coerce')>0)& (pd.to_numeric(df["As_dh"], errors='coerce')>0)
            & (pd.to_numeric(df["X"], errors='coerce')>=17000)& (pd.to_numeric(df["X"], errors='coerce')<17500)
            & (pd.to_numeric(df["Y"], errors='coerce')>=107000)& (pd.to_numeric(df["Y"], errors='coerce')<107500)
            & (pd.to_numeric(df["Z"], errors='coerce')>=2500)& (pd.to_numeric(df["Z"], errors='coerce')<3000)]
    
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

    df['CuT_dh_log'] = np.log(df['CuT_dh'])#stats.zscore(df['CuT_dh'])
    df['CuT_dh_log'] = round(df['CuT_dh_log'],3)
    
    df['Fe_dh_log'] = np.log(df['Fe_dh'])#stats.zscore(df['Fe_dh'])
    df['Fe_dh_log'] = round(df['Fe_dh_log'],3)
    
    df['As_dh_log'] = np.log(df['As_dh'])#stats.zscore(df['As_dh'])
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
    df2['Cu'] = df2['CuT_dh_log_noise']
    df2['Fe'] = df2['Fe_dh_log_noise']


    #df2.groupby(['LITH']).size()

    #df2 = df2.loc[df2['LITH']==31]
    #df2 = df2.reset_index(drop=True)
    n = 100
    m = 50
    xx1 = np.arange(17000, 17500, n).astype('float64')
    yy1 = np.arange(107000, 107500, n).astype('float64')
    zz1 = np.arange(2500, 3000, m).astype('float64')

    blocks = []
    for k in zz1:
        for j in yy1:
            for i in xx1:
                sub_block = df2.loc[(pd.to_numeric(df2["X"], errors='coerce')>=i) & (pd.to_numeric(df2["X"], errors='coerce')<i+n) &
                             (pd.to_numeric(df2["Y"], errors='coerce')>=j) & (pd.to_numeric(df2["Y"], errors='coerce')<j+n)
                             &(pd.to_numeric(df2["Z"], errors='coerce')>=k) & (pd.to_numeric(df2["Z"], errors='coerce')<k+m)]
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
        train = blocks1[i].iloc[list(kf.split(blocks1[i]))[0][0]]
        test = blocks1[i].iloc[list(kf.split(blocks1[i]))[0][1]]
        blocks1_train.append(train)
        blocks1_test.append(test)

    df2_train = pd.concat(blocks1_train)   
    df2_test = pd.concat(blocks1_test)           

    block_idxs1 = np.array(df2_new['blocks'])
    n_blocks = len(df2_new['blocks'].unique())

    # fig = px.scatter_3d(df2_new, x="X",y="Y",z="Z",color="Cu")
    # fig.update_traces(marker_size=2)
    # fig.update_layout(font=dict(size=22))
    # fig.update_layout(scene_aspectmode='data')
    # fig.show()    
    
    fig = px.scatter_3d(df2_new, x="X",y="Y",z="Z",color="Cu")
    fig.update_traces(marker_size=2)
    fig.update_layout(font=dict(size=14))
    fig.update_layout(scene_aspectmode='data')
    # fig.update_layout(scene=dict(
    #     xaxis=dict(showticklabels=False),
    #     yaxis=dict(showticklabels=False),
    #     zaxis=dict(showticklabels=False),
    # ))
    fig.update_layout(scene = dict(
                        xaxis_title='',
                        yaxis_title='',
                        zaxis_title=''))
    fig.update_layout(coloraxis_colorbar_thickness=50) 
    fig.update_layout(coloraxis_colorbar_len=1)  
    fig.show()    

    
    
    

    plt.hist(df2_new.groupby(['blocks']).size(),bins=30,color='b')
    plt.xlabel('Numbers of bore core data for each block', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
        
    # Fe_mean = []
    # Fe_var = []

    
    # for i in blocks1:
    #     Fe_mean.append(np.mean(i['CuT_dh_log_noise']))
    #     Fe_var.append(np.var(i['CuT_dh_log_noise']))

    # plt.figure(figsize=(10, 6))    
    # plt.errorbar(np.arange(0,50,1), Fe_mean[0:50], yerr=Fe_var[0:50], fmt='o', color='black',
    #          ecolor='lightgray', elinewidth=3, capsize=0);    
    # plt.xlabel('Block number',fontsize=14)
    # plt.ylabel('Log Au',fontsize=14)
    # plt.tick_params(axis='both', which='major', labelsize=14)

    
    import pymc3 as pm

    # with pm.Model() as pooled_model:
    #     alpha = pm.Normal('alpha',mu = 0, sd = 1)
    #     beta = pm.Normal('beta',mu = 0, sd = 1)
    #     eps = pm.Uniform('sigma', lower=0, upper=1)
    #     Fe_mean = alpha + beta*df2_new['CuT_dh_log_noise'].values
    #     Fe = pm.Normal('Fe', mu = Fe_mean, sd = eps, observed = df2_new['Fe_dh_log_noise'])
    # with pooled_model:
    #     pooled_trace = pm.sample(2000)
    
    with pm.Model() as pooled_model_HalfCauchy:
        alpha = pm.Normal('alpha',mu = 0, sd = 1)
        beta = pm.Normal('beta',mu = 0, sd = 1)
        eps = pm.HalfCauchy('sigma', 1)
        Fe_mean = alpha + beta*df2_new['CuT_dh_log_noise'].values
        Fe = pm.Normal('Fe', mu = Fe_mean, sd = eps, observed = df2_new['Fe_dh_log_noise'])
    with pooled_model_HalfCauchy:
        pooled_trace_HalfCauchy = pm.sample(2000)
        
        
    # with pm.Model() as unpooled_model:
    #     #mu_alpha = pm.Normal('mu_alpha',mu=0,sd=100)
    #     #sigma_alpha = pm.HalfCauchy('sigma_alpha',2)
        
    #     #mu_beta = pm.Normal('mu_beta',mu=0,sd=100)
    #     #sigma_beta = pm.HalfCauchy('sigma_beta',2)
        
    #     #eps = pm.Uniform('eps',lower=0, upper=100)
    #     eps = pm.Uniform('sigma', lower=0, upper=1)
    #     alpha = pm.Normal('alpha',mu = 0,sd = 1,shape = n_blocks)
    #     beta = pm.Normal('beta',mu = 0,sd = 1, shape = n_blocks)
        
    #     Fe_mean = alpha[block_idxs1] + beta[block_idxs1]*df2_new['CuT_dh_log_noise'].values
    #     Fe = pm.Normal('Fe', mu = Fe_mean, sd = eps,observed = df2_new['Fe_dh_log_noise'])
        
    # with unpooled_model:
    #     unpooled_trace = pm.sample(2000)
             
    with pm.Model() as unpooled_model_HalfCauchy:
        eps = pm.HalfCauchy('sigma', 1)
        alpha = pm.Normal('alpha',mu = 0,sd = 1,shape = n_blocks)
        beta = pm.Normal('beta',mu = 0,sd = 1, shape = n_blocks)
        
        Fe_mean = alpha[block_idxs1] + beta[block_idxs1]*df2_new['CuT_dh_log_noise'].values
        Fe = pm.Normal('Fe', mu = Fe_mean, sd = eps,observed = df2_new['Fe_dh_log_noise'])
    with unpooled_model_HalfCauchy:
        unpooled_trace_HalfCauchy = pm.sample(2000)   
        
    
            
    # with pm.Model() as hirearchical_model_normal:
    #     mu_alpha = pm.Normal('mu_alpha',mu=0,sd=1)
    #     sigma_alpha = pm.Uniform('sigma_alpha', lower=0, upper=1)
        
    #     mu_beta = pm.Normal('mu_beta',mu=0,sd=1)
    #     sigma_beta = pm.Uniform('sigma_beta', lower=0, upper=1)
        
    #     eps = pm.Uniform('sigma', lower=0, upper=1)
        
    #     alpha = pm.Normal('alpha',mu = mu_alpha,sd = sigma_alpha, shape = n_blocks)
    #     beta = pm.Normal('beta',mu = mu_beta,sd = sigma_beta, shape = n_blocks)
        
    #     Fe_mean = alpha[block_idxs1] + beta[block_idxs1]*df2_new['CuT_dh_log_noise'].values
    #     Fe = pm.Normal('Fe', mu = Fe_mean, sd = eps,observed = df2_new['Fe_dh_log_noise'], shape = n_blocks)
        
    # with hirearchical_model_normal:
    #     hirearchical_trace_normal = pm.sample(2000)   
        
    # with pm.Model() as hirearchical_model_exponential:
    #     # mu_alpha = pm.Normal('mu_alpha',mu = 1, sd = 1)
    #     # mu_beta = pm.Normal('mu_beta',mu = 1, sd = 1)
    #     eps = pm.Uniform('sigma', lower=0, upper=1)
        
    #     mu_alpha = pm.Exponential('mu_alpha',lam = 0.1)
    #     mu_beta = pm.Exponential('mu_beta',lam = 0.1)
        
    #     sigma_alpha = pm.Uniform('sigma_alpha', lower=0, upper=1)
    #     sigma_beta = pm.Uniform('sigma_beta', lower=0, upper=1)
        
    #     alpha = pm.Normal('alpha',mu = mu_alpha,sd = sigma_alpha, shape = n_blocks)
    #     beta = pm.Normal('beta',mu = mu_beta,sd = sigma_beta, shape = n_blocks)
        
    #     Fe_mean = alpha[block_idxs1] + beta[block_idxs1]*df2_new['CuT_dh_log_noise'].values
    #     Fe = pm.Normal('Fe', mu = Fe_mean, sd = eps,observed = df2_new['Fe_dh_log_noise'], shape = n_blocks) 
    # with hirearchical_model_exponential:
    #     hirearchical_trace_exponential = pm.sample(2000)  
        
    # with pm.Model() as hirearchical_model_lognormal:
    #     # mu_alpha = pm.Normal('mu_alpha',mu = 0, sd = 1)
    #     # mu_beta = pm.Normal('mu_beta',mu = 0, sd = 1)
    #     # eps = pm.Uniform('sigma', lower=0, upper=1)
        
    #     # sigma_alpha = pm.Uniform('sigma_alpha', lower=0, upper=1)
    #     # sigma_beta = pm.Uniform('sigma_beta', lower=0, upper=1)
        
    #     # alpha = pm.LogNormal('alpha',mu = mu_alpha,sigma = sigma_alpha, shape = n_blocks)
    #     # beta = pm.LogNormal('beta',mu = mu_beta,sigma = sigma_beta, shape = n_blocks)
    #     eps = pm.Uniform('sigma', lower=0, upper=1)
        
    #     mu_alpha = pm.LogNormal('mu_alpha',mu = 1,sigma = 1)
    #     mu_beta = pm.LogNormal('mu_beta',mu = 1,sigma = 1)
        
    #     sigma_alpha = pm.Uniform('sigma_alpha', lower=0, upper=1)
    #     sigma_beta = pm.Uniform('sigma_beta', lower=0, upper=1)
        
    #     alpha = pm.Normal('alpha',mu = mu_alpha,sd = sigma_alpha, shape = n_blocks)
    #     beta = pm.Normal('beta',mu = mu_beta,sd = sigma_beta, shape = n_blocks)
        
    #     Fe_mean = alpha[block_idxs1] + beta[block_idxs1]*df2_new['CuT_dh_log_noise'].values
    #     Fe = pm.Normal('Fe', mu = Fe_mean, sd = eps,observed = df2_new['Fe_dh_log_noise'], shape = n_blocks) 
    # with hirearchical_model_lognormal:
    #     hirearchical_trace_lognormal = pm.sample(2000)  
     
        
     
    with pm.Model() as hirearchical_model_exponential_HalfCauchy:
        mu_alpha = pm.Exponential('mu_alpha',lam = 2)
        mu_beta = pm.Exponential('mu_beta',lam = 2)
        eps = pm.HalfCauchy('sigma', 1)
        
        sigma_alpha = pm.HalfCauchy('sigma_alpha', 1)
        sigma_beta = pm.HalfCauchy('sigma_beta', 1)
        
        alpha = pm.Normal('alpha',mu = mu_alpha,sd = sigma_alpha, shape = n_blocks)
        beta = pm.Normal('beta',mu = mu_beta,sd = sigma_beta, shape = n_blocks)
        
        Fe_mean = alpha[block_idxs1] + beta[block_idxs1]*df2_new['CuT_dh_log_noise'].values
        Fe = pm.Normal('Fe', mu = Fe_mean, sd = eps,observed = df2_new['Fe_dh_log_noise'], shape = n_blocks) 
    with hirearchical_model_exponential_HalfCauchy:
        hirearchical_trace_exponential_HalfCauchy = pm.sample(2000)  
        
    with pm.Model() as hirearchical_model_lognormal_HalfCauchy:
        mu_alpha = pm.LogNormal('mu_alpha',mu = 1,sigma = 1)
        mu_beta = pm.LogNormal('mu_beta',mu = 1,sigma = 1)
        eps = pm.HalfCauchy('sigma', 1)
        
        sigma_alpha = pm.HalfCauchy('sigma_alpha', 1)
        sigma_beta = pm.HalfCauchy('sigma_beta', 1)
        
        alpha = pm.Normal('alpha',mu = mu_alpha,sd = sigma_alpha, shape = n_blocks)
        beta = pm.Normal('beta',mu = mu_beta,sd = sigma_beta, shape = n_blocks)
        
        Fe_mean = alpha[block_idxs1] + beta[block_idxs1]*df2_new['CuT_dh_log_noise'].values
        Fe = pm.Normal('Fe', mu = Fe_mean, sd = eps,observed = df2_new['Fe_dh_log_noise'], shape = n_blocks) 
    with hirearchical_model_lognormal_HalfCauchy:
        hirearchical_trace_lognormal_HalfCauchy = pm.sample(2000)  
        
    with pm.Model() as hirearchical_model_normal_HalfCauchy:
        mu_alpha = pm.Normal('mu_alpha',mu=0,sd=1)
        mu_beta = pm.Normal('mu_beta',mu=0,sd=1)
        eps = pm.HalfCauchy('sigma', 1)
        
        sigma_alpha = pm.HalfCauchy('sigma_alpha', 1)
        sigma_beta = pm.HalfCauchy('sigma_beta', 1)
        
        alpha = pm.Normal('alpha',mu = mu_alpha,sd = sigma_alpha, shape = n_blocks)
        beta = pm.Normal('beta',mu = mu_beta,sd = sigma_beta, shape = n_blocks)
        
        Fe_mean = alpha[block_idxs1] + beta[block_idxs1]*df2_new['CuT_dh_log_noise'].values
        Fe = pm.Normal('Fe', mu = Fe_mean, sd = eps,observed = df2_new['Fe_dh_log_noise'], shape = n_blocks)
    with hirearchical_model_normal_HalfCauchy:
        hirearchical_trace_normal_HalfCauchy = pm.sample(2000)   
           
    # az.plot_trace(hirearchical_trace_normal)
    # az.plot_trace(hirearchical_trace_normal,var_names=['alpha'])
    #pooled_trace_HalfCauchy
    #unpooled_trace_HalfCauchy
    # axes = az.plot_trace(hirearchical_trace_normal_HalfCauchy,var_names=['alpha'])
    # fig = axes.ravel()[0].figure
    # fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Journal paper1\\new figure\\Fig.3-31.png', dpi=300)
    # axes = az.plot_trace(hirearchical_trace_normal_HalfCauchy,var_names=['beta'])
    # fig = axes.ravel()[0].figure
    # fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Journal paper1\\new figure\\Fig.3-32.png', dpi=300)
    # axes = az.plot_trace(hirearchical_trace_normal_HalfCauchy,var_names=['sigma'])
    # fig = axes.ravel()[0].figure
    # fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Journal paper1\\new figure\\Fig.3-33.png', dpi=300)

    # completepooling = az.summary(pooled_trace,round_to=5)
    # nopooling = az.summary(unpooled_trace,round_to=5)
    # partialpooling = az.summary(hirearchical_trace_exponential,round_to=5)
    
    # nopooling[0:220]['r_hat'].mean()
    # nopooling[220:440]['r_hat'].mean()
    
    # partialpooling[0:220]['r_hat'].mean()
    # partialpooling[220:440]['r_hat'].mean()
    
    # az.plot_trace(hirearchical_trace_exponential,figsize=(18,14))
    #az.plot_trace(hirearchical_trace_exponential,combined=False,figsize = (20, 8))

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
            
            
    # hier_a = hirearchical_trace_exponential['alpha'][:].mean(axis=0)[0:100]
    # hier_b = hirearchical_trace_exponential['beta'][:].mean(axis=0)[0:100]
    # indv_a = unpooled_trace['alpha'][:].mean(axis=0)[0:100]
    # indv_b = unpooled_trace['beta'][:].mean(axis=0)[0:100]
    # compl_a = pooled_trace['alpha'][:].mean(axis=0)
    # compl_b = pooled_trace['beta'][:].mean(axis=0)
    
    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(111, xlabel='alpha', ylabel='beta',  
    #                      xlim=(-1, 3), ylim=(-2,2))
    # ax.set_xlabel('alpha',fontsize=20)
    # ax.set_ylabel('beta',fontsize=20)
    # #ax.axhline(y=0,color='k',linewidth=3)
    # for i in range(len(indv_b)):  
    #     ax.arrow(indv_a[i], indv_b[i], hier_a[i] - indv_a[i], hier_b[i] - indv_b[i], 
    #              fc="k", ec="k", length_includes_head=True, alpha= 1 , head_width=.04)    
    # ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.scatter(indv_a,indv_b,c='m', s=50, alpha=1, label = 'no pooling')
    # ax.scatter(hier_a,hier_b, c='b', s=50, alpha=1, label = 'partial pooling')
    # ax.scatter(compl_a,compl_b,c='r',marker='o', s=200, alpha=1, label = 'complete pooling')
    # ax.legend(fontsize=20);
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
    # n=151
    # seq1 = np.linspace(-3,0,6) 
    # pred1 = np.zeros((len(seq1),len(pooled_trace[:][::10])*pooled_trace.nchains))
    # for i, w in enumerate(seq1):
    #     pred1[i] = pooled_trace[:][::10]['alpha'] + pooled_trace[:][::10]['beta']*w
    # seq2 = np.linspace(-3,0,6) 
    # pred2 = np.zeros((len(seq2),len(unpooled_trace[:][::10])*unpooled_trace.nchains))
    # for i, w in enumerate(seq2):
    #     pred2[i] = unpooled_trace[:][::10]['alpha'][:,n] + unpooled_trace[:][::10]['beta'][:,n]*w
    # seq3 = np.linspace(-3,0,6) 
    # pred3 = np.zeros((len(seq3),len(hirearchical_trace_exponential[:][::10])*hirearchical_trace_exponential.nchains))
    # for i, w in enumerate(seq3):
    #     pred3[i] = hirearchical_trace_exponential[:][::10]['alpha'][:,n] + hirearchical_trace_exponential[:][::10]['beta'][:,n]*w
    # #plt.plot(seq1, pred1, '.',color='r')
    # ##################error bar################
    # plt.figure(figsize=(10, 6))
    # x1 = np.linspace(-3,0,6) 
    # y1 = pred1.mean(axis=1)
    # y_min1 = pred1.mean(axis=1) - pred1.min(axis=1)
    # y_max1 = pred1.max(axis=1) - pred1.mean(axis=1)
    # yerr = np.vstack((y_min1,y_max1))
    # plt.errorbar(x1,y1,yerr=yerr,fmt='o',color='c',alpha=0.5,label='complete pooling')

    # x2 = np.linspace(-3,0,6) 
    # y2 = pred2.mean(axis=1)
    # y_min2 = pred2.mean(axis=1) - pred2.min(axis=1)
    # y_max2 = pred2.max(axis=1) - pred2.mean(axis=1)
    # yerr2 = np.vstack((y_min2,y_max2))
    # y_max2 + y_min2
    # plt.errorbar(x2,y2,yerr=yerr2,fmt='o',color='b',alpha=0.5,label='no pooling')
    
    # x3 = np.linspace(-3,0,6) 
    # y3 = pred3.mean(axis=1)
    # y_min3 = pred3.mean(axis=1) - pred3.min(axis=1)
    # y_max3 = pred3.max(axis=1) - pred3.mean(axis=1)
    # y_max3 + y_min3
    # yerr3 = np.vstack((y_min3,y_max3))
    # plt.errorbar(x3,y3,yerr=yerr3,fmt='o',color='r',alpha=0.5,label='partial pooling (hierarchical)')
    # plt.scatter(df2_new[df2_new['blocks']==n]['CuT_dh_log_noise'], df2_new[df2_new['blocks']==n]['Fe_dh_log_noise'],label='bore core data',s=200,color='k',marker = '.')
    # plt.legend(fontsize=14,loc='lower left')
    # plt.xlabel('log Cu',fontsize=14)
    # plt.ylabel('log Fe',fontsize=14)
    # plt.ylim(-3,3)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.title('Blocks No.' + str(df2_new[df2_new['blocks']==n]['blocks'].unique()[0]+1))
    # plt.show()







    #plt.scatter(df2_train[df2_train['elevation_inxs']==1]['CuT_dh_log_noise'], df2_train[df2_train['elevation_inxs']==1]['Fe_dh_log_noise'],label='bore core')
    #plt.plot(seq1, pred1.mean(1), 'k')
    # az.plot_hdi(seq1, pred1.T,color='c')
    # az.plot_hdi(seq2, pred2.T,color='r')
    # az.plot_hdi(seq3, pred3.T,color='g')
    # plt.legend()
    
    # seq = np.linspace(-3,3,20) 
    # pred = np.zeros((len(seq),len(unpooled_trace[:][::10])*unpooled_trace.nchains))
    # for i, w in enumerate(seq):
    #     pred[i] = unpooled_trace[:][::10]['alpha'][:,1] + unpooled_trace[:][::10]['beta'][:,1]*w
    # plt.scatter(df2_new[df2_new['blocks']==1]['CuT_dh_log_noise'], df2_new[df2_new['blocks']==1]['Fe_dh_log_noise'])
    # plt.plot(seq, pred.mean(1), 'k')
    # az.plot_hdi(seq, pred.T)

    # seq1 = np.linspace(-3,3,20) 
    # pred1= np.zeros((len(seq1),len(hirearchical_trace_exponential[:][::10])*hirearchical_trace_exponential.nchains))
    # for i, w in enumerate(seq1):
    #     pred1[i] = hirearchical_trace_exponential[:][::10]['alpha'][:,1] + hirearchical_trace_exponential[:][::10]['beta'][:,1]*w
    # plt.scatter(df2_new[df2_new['blocks']==1]['CuT_dh_log_noise'], df2_new[df2_new['blocks']==1]['Fe_dh_log_noise'])
    # plt.plot(seq1, pred1.mean(1), 'k')
    # az.plot_hdi(seq, pred.T)
    # az.plot_hdi(seq1, pred1.T)
    
    # df_test_label = [val for val in list(df2_train.blocks.unique()) for _ in (0, 1)]

    # df_label = [val for val in list(df2_train.blocks.unique()) for _ in (0, 1)]


    
    selection2 = [34,50,130,163,185,210]
    fig,axis = plt.subplots(2,3,figsize=(20,10),sharey=True,sharex=False);
    axis = axis.ravel()
    for i,c in enumerate(selection2):
        c_data = df2_new.loc[df2_new.blocks==c]
        c_data = c_data.reset_index(drop=True)
        xvals = np.linspace(c_data['CuT_dh_log_noise'].min()-1,c_data['CuT_dh_log_noise'].max()+1)
        #A = np.vstack([c_data['CuT_dh_log_noise'], np.ones(len(c_data['CuT_dh_log_noise']))]).T
        #m,d = np.linalg.lstsq(A, np.array(c_data['Fe_dh_log_noise']), rcond=None)[0]
        axis[i].set_title('Block No.' + str(c+1),fontsize=14)
        axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=200,label = 'bore core data')
        axis[i].plot(xvals,pooled_trace_HalfCauchy['alpha'][:].mean()+pooled_trace_HalfCauchy['beta'][:].mean()*xvals,'b',alpha=1,lw=3.,label='complete pooling')
        axis[i].plot(xvals,unpooled_trace_HalfCauchy['alpha'][:,c].mean()+unpooled_trace_HalfCauchy['beta'][:,c].mean()*xvals,'r',alpha=1,lw=3.,label='no pooling')
        axis[i].plot(xvals,hirearchical_trace_normal_HalfCauchy['alpha'][:,c].mean()+hirearchical_trace_normal_HalfCauchy['beta'][:,c].mean()*xvals,'c',alpha=1,lw=3.,label='partial pooling')
        
        axis[i].set_xlim(c_data['CuT_dh_log_noise'].min()-1,c_data['CuT_dh_log_noise'].max()+1)
        axis[i].set_ylim(-1,3)
        if i>2:
            axis[i].set_xlabel('Cu (wt%) log scale',fontsize=14)
        if i%3==0:
            axis[i].set_ylabel('Fe (wt%) log scale',fontsize=14)
        if i==5:
            axis[i].legend(loc='lower right',prop={'size':14})
            
    fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Journal paper1\\new figure\\Fig.5-escondia.png', format='png', dpi=300)
    # for n in  [34,50,130,163,185,210]:
    #     normal_betarange = hirearchical_trace_normal_HalfCauchy['beta'][:,n].max() - hirearchical_trace_normal_HalfCauchy['beta'][:,n].min()
    #     exponential_betarange = hirearchical_trace_exponential_HalfCauchy['beta'][:,n].max() - hirearchical_trace_exponential_HalfCauchy['beta'][:,n].min()
    #     lognormal_betarange = hirearchical_trace_lognormal_HalfCauchy['beta'][:,n].max() - hirearchical_trace_lognormal_HalfCauchy['beta'][:,n].min()
    #     #print(normal_betarange,exponential_betarange,lognormal_betarange)
    #     print(100*(exponential_betarange - normal_betarange)/normal_betarange,100*(lognormal_betarange - normal_betarange)/normal_betarange)
        

    selection2 = [34,50,130,163,185,210,34,50,130,163,185,210]
    fig,axis = plt.subplots(2,6,figsize=(30,8),sharey=True,sharex=False);
    axis = axis.ravel()
    for i,c in enumerate(selection2):
        c_data = df2_new.loc[df2_new.blocks==c]
        c_data = c_data.reset_index(drop=True)
        xvals = np.linspace(c_data['CuT_dh_log_noise'].min()-1,c_data['CuT_dh_log_noise'].max()+1)
        if i<6: 
            axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=200,label = 'bore core')  
            axis[i].set_title('Block No.' + str(c+1),fontsize=14)
            num=1
            for a_val, b_val in zip(unpooled_trace_HalfCauchy['alpha'][:,c],unpooled_trace_HalfCauchy['beta'][:,c]):  
                if num==1:
                    axis[i].plot(xvals,a_val+b_val*xvals,'r',label = 'no pooling')
                    num+=1
                else:
                    axis[i].plot(xvals,a_val+b_val*xvals,'r',alpha=.01)

            if i ==6:
                axis[i].set_ylabel('log Fe (wt)',fontsize=14) 
        else:
            axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=200,label = 'bore core')
            axis[i].set_title('Block No.' + str(c+1),fontsize=14)
            num=1
            for a_val, b_val in zip(hirearchical_trace_normal_HalfCauchy['alpha'][:,c],hirearchical_trace_normal_HalfCauchy['beta'][:,c]):
                if num==1:
                    axis[i].plot(xvals,a_val+b_val*xvals,'c',label = 'partial pooling')
                    num+=1
                else:
                    axis[i].plot(xvals,a_val+b_val*xvals,'c',alpha=.01)
            axis[i].set_xlabel('log Cu (wt)',fontsize=14)
            if i ==0:
                axis[i].set_ylabel('log Fe (wt)',fontsize=14)
        axis[i].legend(loc='lower right',fontsize=14)
        axis[i].tick_params(axis='both', which='major', labelsize=14) 
        axis[0].set_ylabel('log Fe (wt)',fontsize=14)
        axis[6].set_ylabel('log Fe (wt)',fontsize=14)
        fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Journal paper1\\new figure\\Fig.7-escondia.png', dpi=300)  
    
    
    
    selection2 = [34,50,130,163,185,210]
    fig,axis = plt.subplots(2,3,figsize=(20,10),sharey=True,sharex=False);
    axis = axis.ravel()
    for i,c in enumerate(selection2):
        if i<3: 
            axis[i].hist(unpooled_trace_HalfCauchy['alpha'][:,c],bins=50,color='r',lw=2,histtype='step',label = 'Alpha (no pooling)')  
            axis[i].hist(hirearchical_trace_normal_HalfCauchy['alpha'][:,c],bins=50,color='k',lw=2,histtype='step',label = 'Alpha (partial pooling)')  
            axis[i].hist(unpooled_trace_HalfCauchy['beta'][:,c],bins=50,color='c',lw=2,histtype='step',label = 'Beta (no pooling)')  
            axis[i].hist(hirearchical_trace_normal_HalfCauchy['beta'][:,c],bins=50,color='m',lw=2,histtype='step',label = 'Beta (partial pooling)')  
            axis[i].set_title('Block No.' + str(c+1),fontsize=14)
            if i==0:
                axis[i].set_ylabel('Frequency',fontsize=14)
        else:
            axis[i].hist(unpooled_trace_HalfCauchy['alpha'][:,c],bins=50,color='r',lw=2,histtype='step',label = 'Alpha (no pooling)')  
            axis[i].hist(hirearchical_trace_normal_HalfCauchy['alpha'][:,c],bins=50,color='k',lw=2,histtype='step',label = 'Alpha (partial pooling)')  
            axis[i].hist(unpooled_trace_HalfCauchy['beta'][:,c],bins=50,color='c',lw=2,histtype='step',label = 'Beta (no pooling)')  
            axis[i].hist(hirearchical_trace_normal_HalfCauchy['beta'][:,c],bins=50,color='m',lw=2,histtype='step',label = 'Beta (partial pooling)')  
            axis[i].set_title('Block No.' + str(c+1),fontsize=14)
            axis[i].set_xlabel('Model parameter',fontsize=14)
            if i ==3:
                axis[i].set_ylabel('Frequency',fontsize=14)
            elif i==5:
                axis[i].legend(loc='center left',bbox_to_anchor=(1, 0.5),fontsize=14)
        axis[i].tick_params(axis='both', which='major', labelsize=14) 
    #fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Journal paper1\\new figure\\Fig.9-escondia.png', bbox_inches='tight', dpi=300) 
    
    
    #pred_unpooled_list = []
    pred_normal_pooled_list = []
    pred_exp_pooled_list = []
    pred_lognormal_pooled_list = []
    selection2 = [34,50,130,163,185,210]
    for i,c in enumerate(selection2):
        c_data = df2_new.loc[df2_new.blocks==c]
        c_data = c_data.reset_index(drop=True)
        xvals = np.linspace(c_data['CuT_dh_log_noise'].min()-1,c_data['CuT_dh_log_noise'].max()+1,5)
        #pred_unpooled = []
        pred_normal_pooled = []
        pred_exp_pooled = []
        pred_lognormal_pooled = []
        for x in xvals:
            #predicted_unpooled  = unpooled_trace_HalfCauchy['alpha'][:,c] + unpooled_trace_HalfCauchy['beta'][:,c]*x
            predicted_normal_pooled  = hirearchical_trace_normal_HalfCauchy['alpha'][:,c] + hirearchical_trace_normal_HalfCauchy['beta'][:,c]*x
            predicted_exp_pooled = hirearchical_trace_exponential_HalfCauchy['alpha'][:,c] + hirearchical_trace_exponential_HalfCauchy['beta'][:,c]*x
            predicted_lognormal_pooled = hirearchical_trace_lognormal_HalfCauchy['alpha'][:,c] + hirearchical_trace_lognormal_HalfCauchy['beta'][:,c]*x
            #pred_unpooled.append(predicted_unpooled)
            pred_normal_pooled.append(predicted_normal_pooled)
            pred_exp_pooled.append(predicted_exp_pooled)
            pred_lognormal_pooled.append(predicted_lognormal_pooled)
        #pred_unpooled_list.append(pred_unpooled)
        pred_normal_pooled_list.append(pred_normal_pooled)
        pred_exp_pooled_list.append(pred_exp_pooled)
        pred_lognormal_pooled_list.append(pred_lognormal_pooled)
    # for i in range(5):
    #     #print(round((pred_normal_pooled_list[5][i].max() - pred_normal_pooled_list[5][i].min()),2))
    #     #print(round((pred_exp_pooled_list[5][i].max() - pred_exp_pooled_list[5][i].min()),2))
    #     print(round((pred_lognormal_pooled_list[5][i].max() - pred_lognormal_pooled_list[5][i].min()),2))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    nopooled_betarange = unpooled_trace_HalfCauchy['beta'][:,216].max() - unpooled_trace_HalfCauchy['beta'][:,216].min()
    partialpooled_betarange = hirearchical_trace_normal_HalfCauchy['beta'][:,216].max() - hirearchical_trace_normal_HalfCauchy['beta'][:,216].min()
    (nopooled_betarange - partialpooled_betarange)/nopooled_betarange
    
    # for n in selection2:
    #     normal_betarange = hirearchical_trace_normal_HalfCauchy['beta'][:,n].max() - hirearchical_trace_normal_HalfCauchy['beta'][:,n].min()
    #     nopooling_betarange = unpooled_trace_HalfCauchy['beta'][:,n].max() - unpooled_trace_HalfCauchy['beta'][:,n].min()
    #     print(100*(normal_betarange - nopooling_betarange)/nopooling_betarange)
        
    # selection2 = [2,34,82,96,159,216,2,34,82,96,159,216,2,34,82,96,159,216]
    # fig,axis = plt.subplots(3,6,figsize=(30,8),sharey=True,sharex=False);
    # axis = axis.ravel()
    # for i,c in enumerate(selection2):
    #     c_data = df2_new.loc[df2_new.blocks==c]
    #     c_data = c_data.reset_index(drop=True)
    #     xvals = np.linspace(c_data['CuT_dh_log_noise'].min()-1,c_data['CuT_dh_log_noise'].max()+1)
    #     if i<6:
    #         axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=200,label = 'bore core')  
    #         axis[i].set_title('Block No.' + str(c+1),fontsize=14)
    #         num=1
    #     #axis[i].plot(xvals,hirearchical_trace['alpha'][1000:,c_index].mean()+hirearchical_trace['beta'][1000:,c_index].mean()*xvals,'r',alpha=1,lw=1.,label='hirearchical')
    #         for a_val, b_val in zip(hirearchical_trace_normal['alpha'][:,c],hirearchical_trace_normal['beta'][:,c]):  
    #             if num==1:
    #                 axis[i].plot(xvals,a_val+b_val*xvals,'r',label = 'Normal')
    #                 num+=1
    #             else:
    #                 axis[i].plot(xvals,a_val+b_val*xvals,'r',alpha=.01)
    #         if i==6:
    #             axis[i].set_ylabel('log Fe',fontsize=14)
    #     elif i>=6 and i<=11:
    #         axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=200,label = 'bore core')
    #         axis[i].set_title('Block No.' + str(c+1),fontsize=14)
    #         num=1
    #         for a_val, b_val in zip(hirearchical_trace_exponential['alpha'][:,c],hirearchical_trace_exponential['beta'][:,c]):
    #             if num==1:
    #                 axis[i].plot(xvals,a_val+b_val*xvals,'c',label = 'Exponential')
    #                 num+=1
    #             else:
    #                 axis[i].plot(xvals,a_val+b_val*xvals,'c',alpha=.01)
    #         if i==6:
    #             axis[i].set_ylabel('log Fe',fontsize=14)
                
    #     else:
    #         axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=200,label = 'bore core')
    #         axis[i].set_title('Block No.' + str(c+1),fontsize=14)
    #         axis[i].set_xlabel('log Cu',fontsize=14)
    #         num=1
    #         for a_val, b_val in zip(hirearchical_trace_lognormal['alpha'][:,c],hirearchical_trace_lognormal['beta'][:,c]):
    #             if num==1:
    #                 axis[i].plot(xvals,a_val+b_val*xvals,'m',label = 'Lognormal')
    #                 num+=1
    #             else:
    #                 axis[i].plot(xvals,a_val+b_val*xvals,'m',alpha=.01)
    #         if i==12:
    #             axis[i].set_ylabel('log Fe',fontsize=14)
    #     axis[i].legend(loc='lower right',fontsize=14)
    #     axis[i].tick_params(axis='both', which='major', labelsize=14)  
        

    list_exp_nor = []
    list_lognor_nor = []
    list_exp_betarange = []
    list_lognor_betarange = []
    list_normal_betarange = []
    for n in selection2:
        normal_betarange = hirearchical_trace_normal_HalfCauchy['beta'][:,n].max() - hirearchical_trace_normal_HalfCauchy['beta'][:,n].min()
        exponential_betarange = hirearchical_trace_exponential_HalfCauchy['beta'][:,n].max() - hirearchical_trace_exponential_HalfCauchy['beta'][:,n].min()
        lognormal_betarange = hirearchical_trace_lognormal_HalfCauchy['beta'][:,n].max() - hirearchical_trace_lognormal_HalfCauchy['beta'][:,n].min()
        list_exp_betarange.append(exponential_betarange)
        list_lognor_betarange.append(lognormal_betarange)
        list_normal_betarange.append(normal_betarange)
        list_exp_nor.append(100*(exponential_betarange - normal_betarange)/normal_betarange)
        list_lognor_nor.append(100*(lognormal_betarange - normal_betarange)/normal_betarange)


    label1 = ['No.35','No.51','No.131','No.164','No.199','No.211']
    fig, axis = plt.subplots(1,1,figsize=(12,6))
    x = np.arange(6)
    bar1 = axis.bar(x-0.2,list_normal_betarange,color='m',width=0.2,label = 'Normal prior')        
    bar2 = axis.bar(x,list_exp_betarange,color='c',width=0.2,label = 'Exponential prior')      
    bar3 = axis.bar(x+0.2,list_lognor_betarange,color='b',width=0.2,label = 'Log normal prior')     
    axis.set_xticks([0,1,2,3,4,5],label1,fontsize=14)
    axis.tick_params(axis='both', which='major', labelsize=14)
    axis.set_xlabel('Block',fontsize=18)
    axis.set_ylabel('Beta',fontsize=18)
    axis.legend(fontsize=18)
    #fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Journal paper1\\new figure\\Fig.11-escondia.png',dpi=300) 
    
    
    idx1 = [] 
    idx2 = []
    list2 = []
    for n in np.arange(len(blocks1)):
        normal_betarange1 = hirearchical_trace_normal_HalfCauchy['beta'][:,n].max() - hirearchical_trace_normal_HalfCauchy['beta'][:,n].min()
        nopooling_betarange = unpooled_trace_HalfCauchy['beta'][:,n].max() - unpooled_trace_HalfCauchy['beta'][:,n].min()
        variance = 100*(normal_betarange1 - nopooling_betarange)/normal_betarange1
        list2.append(variance)
        if variance > 0:
            idx1.append(n)
        else:
            idx2.append(n)

    for i in np.arange(len(blocks1)):
        blocks1[i]['variance value'] = list2[i]
    
    block_lowvariance = []
    for i in idx1:
        block_lowvariance.append(blocks1[i])
    data_lowvariance = pd.concat(block_lowvariance)       
    data_lowvariance['variance'] = 'low'

    block_highvariance = []
    for i in idx2:
        block_highvariance.append(blocks1[i])
    data_highvariance = pd.concat(block_highvariance)      
    data_highvariance['variance'] = 'high'
        
    # fig = px.scatter_3d(data_lowvariance, x="X",y="Y",z="Z",color="CU_wt_log_noise")
    # fig.update_traces(marker_size=2)
    # fig.update_layout(font=dict(size=14))
    # fig.show()          
        

    data_variance = pd.concat([data_lowvariance, data_highvariance])
   
    
    selection2 = [34,50,130,163,185,210]
    from sklearn.linear_model import LinearRegression
    r_squared_list = []
    samplesize_list = []
    for i in selection2:
        model = LinearRegression()
        df2_sub = df2_new[df2_new['blocks']==i]
        x,y = np.array(df2_sub["CuT_dh_log_noise"]), np.array(df2_sub["Fe_dh_log_noise"])
        model.fit(x.reshape(-1,1), y)
        r_squared = model.score(x.reshape(-1,1), y)
        r_squared_list.append(r_squared)
        samplesize_list.append(len(df2_sub))
        
    fig, ax = plt.subplots(1,1,figsize=(14,10))    
    ax.scatter(samplesize_list,r_squared_list,marker='o',color='r',s=200)
    ax.set_xlabel('Sample size for each block',fontsize=22)
    ax.set_ylabel('$R^2$',fontsize=22)
    ax.set_xlim(0,180)
    ax.set_ylim(0,0.4)
    ax.tick_params(axis='both', which='major', labelsize=16)
    for i, txt in enumerate(selection2):
        ax.annotate('block No.' +str(txt+1), (samplesize_list[i], r_squared_list[i]),xytext=(samplesize_list[i]+2, r_squared_list[i]-0.005),fontsize=16)
    fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Journal paper1\\new figure\\Fig.14-escondia.png', bbox_inches='tight', dpi=300)   
      
    
    
    
    for n in selection2:
        normal_betarange1 = hirearchical_trace_normal_HalfCauchy['beta'][:,n].max() - hirearchical_trace_normal_HalfCauchy['beta'][:,n].min()
        nopooling_betarange = unpooled_trace_HalfCauchy['beta'][:,n].max() - unpooled_trace_HalfCauchy['beta'][:,n].min()
        variance = 100*(normal_betarange1 - nopooling_betarange)/normal_betarange1

    


    idx1 = [] 
    idx2 = []
    list2 = []
    for n in np.arange(len(blocks1)):
        normal_betarange1 = hirearchical_trace_normal_HalfCauchy['beta'][:,n].max() - hirearchical_trace_normal_HalfCauchy['beta'][:,n].min()
        nopooling_betarange = unpooled_trace_HalfCauchy['beta'][:,n].max() - unpooled_trace_HalfCauchy['beta'][:,n].min()
        variance = 100*(normal_betarange1 - nopooling_betarange)/normal_betarange1
        list2.append(variance)
        if variance > -10:
            idx1.append(n)
        else:
            idx2.append(n)
    
    for i in np.arange(len(blocks1)):
        blocks1[i]['variance value'] = list2[i]
    
    block_lowvariance = []
    for i in idx1:
        block_lowvariance.append(blocks1[i])
    data_lowvariance = pd.concat(block_lowvariance)       
    data_lowvariance['variance'] = 'low'
    
    block_highvariance = []
    for i in idx2:
        block_highvariance.append(blocks1[i])
    data_highvariance = pd.concat(block_highvariance)      
    data_highvariance['variance'] = 'high'
        
    data_variance = pd.concat([data_lowvariance, data_highvariance])
    
    


   ###########calculate mean error difference between no pool and partial pool
    list_completepool_mean = []
    list_nopool_mean = []
    list_partialpool_mean = []
    for i in range(len(blocks1)):
         list_completepool_mean.append(pooled_trace_HalfCauchy['beta'][:].mean())
         list_nopool_mean.append(unpooled_trace_HalfCauchy['beta'][:,i].mean())
         list_partialpool_mean.append(hirearchical_trace_normal_HalfCauchy['beta'][:,i].mean())

    ############sample size and R square#############
    from sklearn.linear_model import LinearRegression
    r_squared_list = []
    samplesize_list = []
    Cu_mean_list = []
    Fe_mean_list = []
    Cu_var_list = []
    Fe_var_list = []
    mean = pd.DataFrame()
    for i in range(len(blocks1)):
        model = LinearRegression()
        df2_sub = df2_new[df2_new['blocks']==i]
        x,y = np.array(df2_sub["CuT_dh_log_noise"]), np.array(df2_sub["Fe_dh_log_noise"])
        model.fit(x.reshape(-1,1), y)
        r_squared = model.score(x.reshape(-1,1), y)
        r_squared_list.append(r_squared)
        samplesize_list.append(len(df2_sub))
        Cu_mean_list.append(df2_sub['CuT_dh_log_noise'].mean())
        Fe_mean_list.append(df2_sub['Fe_dh_log_noise'].mean())
        Cu_var_list.append(df2_sub['CuT_dh_log_noise'].var())
        Fe_var_list.append(df2_sub['Fe_dh_log_noise'].var())
    mean['sample size'] = samplesize_list    
    mean['R^2'] = r_squared_list
    mean['block'] = np.arange(0,len(blocks1),1)
    mean['mean Cu'] = Cu_mean_list
    mean['mean Fe'] = Fe_mean_list
    mean['var Cu'] = Cu_var_list
    mean['var Fe'] = Fe_var_list

    # fig, ax = plt.subplots(1,1,figsize=(14,10))    
    # ax.scatter(mean_largedifference['sample size'],mean_largedifference['var Cu'],marker='o',color='r',s=200,label='more shrinkage blocks')
    # ax.scatter(mean_smalldifference['sample size'],mean_smalldifference['var Cu'],marker='o',color='black',s=200,label='less shrinkage blocks')
    # #ax.set_xlabel('Sample size for each block',fontsize=22)
    # #ax.set_ylabel('$R^2$',fontsize=22)
    # ax.legend(fontsize=22)
    # # ax.set_xlim(0,180)
    # # ax.set_ylim(0,0.5)
    # fig = px.scatter_3d(mean, x="var Cu",y="var Fe",z="sample size",color="shrinkage")
    # fig.update_traces(marker_size=8)
    # fig.update_layout(font=dict(size=14))
    # #fig.update_layout(scene_aspectmode='data')
    # fig.show()    



        
    fig, ax = plt.subplots(1,1,figsize=(14,10))    
    ax.scatter(samplesize_list,r_squared_list,marker='o',color='r',s=200)
    ax.set_xlabel('Sample size for each block',fontsize=22)
    ax.set_ylabel('$R^2$',fontsize=22)
    ax.set_xlim(0,180)
    ax.set_ylim(0,0.4)
    ax.tick_params(axis='both', which='major', labelsize=16)
    for i, txt in enumerate(selection2):
        ax.annotate('block No.' +str(txt+1), (samplesize_list[i], r_squared_list[i]),xytext=(samplesize_list[i]+2, r_squared_list[i]-0.005),fontsize=16)
    
    
    selection3 = [178,202,201,153,34,45]
    fig,axis = plt.subplots(2,3,figsize=(20,10),sharey=True,sharex=False);
    axis = axis.ravel()
    for i,c in enumerate(selection3):
        c_data = df2_new.loc[df2_new.blocks==c]
        c_data = c_data.reset_index(drop=True)
        xvals = np.linspace(c_data['CuT_dh_log_noise'].min()-1,c_data['CuT_dh_log_noise'].max()+1)
        #A = np.vstack([c_data['CuT_dh_log_noise'], np.ones(len(c_data['CuT_dh_log_noise']))]).T
        #m,d = np.linalg.lstsq(A, np.array(c_data['Fe_dh_log_noise']), rcond=None)[0]
        axis[i].set_title('Block No.' + str(c+1),fontsize=14)
        axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=200,label = 'bore core data')
        axis[i].plot(xvals,pooled_trace_HalfCauchy['alpha'][:].mean()+pooled_trace_HalfCauchy['beta'][:].mean()*xvals,'b',alpha=1,lw=3.,label='complete pooling')
        axis[i].plot(xvals,unpooled_trace_HalfCauchy['alpha'][:,c].mean()+unpooled_trace_HalfCauchy['beta'][:,c].mean()*xvals,'r',alpha=1,lw=3.,label='no pooling')
        axis[i].plot(xvals,hirearchical_trace_normal_HalfCauchy['alpha'][:,c].mean()+hirearchical_trace_normal_HalfCauchy['beta'][:,c].mean()*xvals,'c',alpha=1,lw=3.,label='partial pooling')
        
        axis[i].set_xlim(c_data['CuT_dh_log_noise'].min()-1,c_data['CuT_dh_log_noise'].max()+1)
        axis[i].set_ylim(-1,3)
        if i>2:
            axis[i].set_xlabel('Cu (wt%) log scale',fontsize=14)
        if i%3==0:
            axis[i].set_ylabel('Fe (wt%) log scale',fontsize=14)
        if i==5:
            axis[i].legend(loc='lower right',prop={'size':14})


    selection3 =[16,204,76,25,164,182]
    #[106,87,153,33,50,179] #[13,193,82,29,120,144]#[31,24,164,201,7,109]#[71,2,202,76,5,170]#[45,34,39,178,16,25] 
    fig,axis = plt.subplots(2,3,figsize=(20,10),sharey=True,sharex=False);
    axis = axis.ravel()
    for i,c in enumerate(selection3):
        c_data = df2_new.loc[df2_new.blocks==c]
        c_data = c_data.reset_index(drop=True)
        axis[i].hist(c_data['LITH'],label=str(c+1))
        axis[i].legend()

    ##########shrinkage#########
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    #ax.scatter(pooled_trace_HalfCauchy['alpha'].mean(),pooled_trace_HalfCauchy['beta'].mean(),color='red', s=300,label = 'complete-pooling')
    ax.scatter(unpooled_trace_HalfCauchy['alpha'].mean(axis=0),unpooled_trace_HalfCauchy['beta'].mean(axis=0),color='b', s=50,label = 'no-pooling')
    ax.scatter(hirearchical_trace_normal_HalfCauchy['alpha'].mean(axis=0),hirearchical_trace_normal_HalfCauchy['beta'].mean(axis=0), c='m', s=50,  label = 'partial-pooling')
    for i in range(len(unpooled_trace_HalfCauchy['alpha'].mean(axis=0))):  
        ax.arrow(unpooled_trace_HalfCauchy['alpha'].mean(axis=0)[i], unpooled_trace_HalfCauchy['beta'].mean(axis=0)[i], hirearchical_trace_normal_HalfCauchy['alpha'].mean(axis=0)[i]\
                 - unpooled_trace_HalfCauchy['alpha'].mean(axis=0)[i], hirearchical_trace_normal_HalfCauchy['beta'].mean(axis=0)[i] - unpooled_trace_HalfCauchy['beta'].mean(axis=0)[i], 
                 fc="k", ec="k", length_includes_head=True,head_width=.03)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel('alpha',fontsize=16)
    ax.set_ylabel('beta',fontsize=16)
    ax.legend(fontsize=16)
    fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Journal paper1\\new figure\\Fig.16-escondida.png', bbox_inches='tight', dpi=300)   
    
    
    
    
    distance = []
    lith_classes = []
    for i in range(len(unpooled_trace_HalfCauchy['alpha'].mean(axis=0))):  
        x = abs(hirearchical_trace_normal_HalfCauchy['alpha'].mean(axis=0)[i] - unpooled_trace_HalfCauchy['alpha'].mean(axis=0)[i])
        y = abs(hirearchical_trace_normal_HalfCauchy['beta'].mean(axis=0)[i] - unpooled_trace_HalfCauchy['beta'].mean(axis=0)[i])
        distance.append(np.sqrt(x**2+y**2))
        lith_classes.append(len(df2_new[df2_new['blocks']==i]['LITH'].unique()))
    mean['distance']  = distance
    mean['lith categories'] = lith_classes
    plt.hist(mean[mean['lith categories']==1]['distance'],bins=100)


    selection3 =[1,2,3,4,5,6]
    #[106,87,153,33,50,179] #[13,193,82,29,120,144]#[31,24,164,201,7,109]#[71,2,202,76,5,170]#[45,34,39,178,16,25] 
    fig,axis = plt.subplots(2,3,figsize=(20,10),sharey=True,sharex=False);
    axis = axis.ravel()
    for i,c in enumerate(selection3):
        axis[i].hist(mean[mean['lith categories']==c]['distance'],bins=100,label='total lith categories are '+str(c))
        axis[i].legend(fontsize=16)
        axis[i].tick_params(axis='both', which='major', labelsize=16)
        axis[i].set_xlabel('shrinkage distance',fontsize=16)
        axis[i].set_ylabel('frequency',fontsize=16)
        axis[i].set_xlim(0,1.8)
    
    fig = px.scatter_3d(mean, x="sample size",y="lith categories",z="distance")
    fig.update_traces(marker_size=8)
    fig.update_layout(font=dict(size=16))
    #fig.update_layout(scene_aspectmode='data')
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(
                    text='sample size',
                    font=dict(
                        size=30  # Increase the font size here as per your requirement
                    )
                )
            ),
            yaxis=dict(
                title=dict(
                    text='lithology categories',
                    font=dict(
                        size=30  # Increase the font size here as per your requirement
                    )
                )
            ),
            zaxis=dict(
                title=dict(
                    text='shrinkage degree',
                    font=dict(
                        size=30  # Increase the font size here as per your requirement
                    )
                )
            )
        )
    )
    fig.show()   
    fig.write_image('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Journal paper1\\figure\\Fig.18.png',width=1600,height=1600,scale=2,engine = 'kaleido')   
    
    var_betweengroup = []
    for i in range(len(unpooled_trace_HalfCauchy['alpha'].mean(axis=0))):  
        var_betweengroup.append(df2_new[df2_new['blocks']==i]['Fe_dh_log_noise'].var())

    # mean_cu = []
    # mean_fe = []
    # for i in range(n_blocks):
    #     mean_cu.append(blocks1[i]['CuT_dh_log_noise'].mean())
    #     mean_fe.append(blocks1[i]['Fe_dh_log_noise'].mean())

    # from sklearn.linear_model import LinearRegression
    # r_squared_list = []
    # for i in range(n_blocks):
    #     model = LinearRegression()
    #     x,y = np.array(blocks1[i]['Fe_dh_log_noise']), np.array(blocks1[i]['CuT_dh_log_noise'])
    #     model.fit(x.reshape(-1,1), y)
    #     r_squared = model.score(x.reshape(-1,1), y)
    #     if r_squared >0:
    #         r_squared_list.append(r_squared)
    #         blocks1[i]['r2'] = r_squared
    #     else:
    #         r_squared_list.append(0)


    
    
