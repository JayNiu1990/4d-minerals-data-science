if __name__ == '__main__':
    import warnings
    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import pymc3 as pm
    import pandas as pd
    import theano.tensor as tt
    import random
    import os
    pseudo_truck=[]
    files = os.listdir('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\individual_truck_grade')
    for i in files:
        pseudo_truck.append(pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\individual_truck_grade\\"+str(i),sep=" "))
    
    pseudo_truck1 = []
    for j in pseudo_truck:
        for i in range(len(j)):
            pseudo_truck1.append(j['Individual'][i].split(','))
    
    pseudo_truck_new = []
    for i in pseudo_truck1:
        if len(i)>75:
            pseudo_truck2 = []
            for j in i:
                pseudo_truck2.append(float(j))
            pseudo_truck_new.append(pseudo_truck2)    
    
    df = pd.DataFrame(pseudo_truck_new)
    
    
    #df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\2min_6min_time_elapse_individual_truck_grade_Oct21_Mar22.csv")
    counts = df.notnull().sum(axis=1)
    # 
    # df['index'] = df.index
    # a = df.iloc[[0]].values
    # a = list(a)
    # a1 = [x for x in a[0] if np.isnan(x) == False]
    # a2 = pd.DataFrame(random.sample(a1,int(counts[0]*0.05))).T
    df_partial = []
    for i in range(len(df)):
        a = list(df.iloc[[i]].values)
        a1 = [x for x in a[0] if np.isnan(x) == False]
        #a2 = pd.DataFrame(random.sample(a1,int(counts[i]*0.25))).T
        a2 = pd.DataFrame(a1[0:int(counts[i]*0.25)]).T
        df_partial.append(a2)
        
    df_partial = pd.concat(df_partial)
    df_partial['index'] = range(0, len(df_partial))
    df_partial = df_partial.set_index('index')
    
    df["average_grade"] = df.mean(axis=1)
    df_partial["average_grade"] = df_partial.mean(axis=1)

    df_train_x = df_partial["average_grade"][0:201] 
    df_train_y = df["average_grade"][0:201] 

    # plt.scatter(df_partial["average_grade"][0:100],df["average_grade"][0:100])
    # plt.xlabel('mean grade on 5% truck load',fontsize=12)
    # plt.ylabel('mean grade on entire truck load',fontsize=12)
    
    with pm.Model() as pooled_model_HalfCauchy:
        alpha = pm.Normal('alpha',mu = 0, sd = 1)
        beta = pm.Normal('beta',mu = 0, sd = 1)
        eps = pm.Uniform('sigma', lower=0, upper=1)
        pseduo_truck_mean = alpha + beta*df_train_x.values
        pseduo_truck = pm.Normal('pseduo_truck', mu = pseduo_truck_mean, sd = eps, observed = df_train_y)
    with pooled_model_HalfCauchy:
        pooled_trace_HalfCauchy = pm.sample(2000)

    xvals = np.linspace(df_train_x.min(),df_train_x.max())
    plt.scatter(df_train_x,df_train_y,color='k',marker='.',s=4,label = 'bore core data')
    
    
    for a_val, b_val in zip(pooled_trace_HalfCauchy['alpha'][:],pooled_trace_HalfCauchy['beta'][:]):  
        plt.plot(xvals,a_val+b_val*xvals,'r',alpha=.01)

    df_test_x = pd.DataFrame(np.array(df_partial["average_grade"][201:301]),columns=['average grade for truck sensor'])
    df_test_y = pd.DataFrame(np.array(df["average_grade"][201:301]),columns=['average grade for pseduo truck'])
    
    pred = []
    for i in df_test_x['average grade for truck sensor']:
        pred_sub = []
        for a_val, b_val in zip(pooled_trace_HalfCauchy['alpha'][:],pooled_trace_HalfCauchy['beta'][:]):  
            pred_sub.append(a_val+b_val*i)
        pred.append(pred_sub)
    #plt.hist(pred[0],bins=100)
    pred = pd.DataFrame(pred)
    #pred['mean pred'] = pred.mean(axis=1)
    # pred['average grade of truck sensor'] = df_test_x['average grade for truck sensor']
    # pred['average grade of pseduo truck'] = df_test_y['average grade for pseduo truck']

    import scipy   
    from scipy.stats import norm
    loc,scale = scipy.stats.distributions.norm.fit(pred[0])
    pdf = 1 - scipy.stats.norm.cdf(0.6,loc=loc,scale=scale)

    


    mu,std = norm.fit(pred[0:1])
    xmin, xmax = np.array(pred[0:1]).min(),np.array(pred[0:1]).max()
    x = np.linspace(xmin, xmax, 100)
    pdf = norm.pdf(x, mu, std)
    plt.plot(x,pdf*std,label = "normal fit")
    #plt.hist(np.array(pred[0:1])[0],density=True, bins=100)
    plt.xlabel("grade from mcmc")
    plt.ylabel("density")
    plt.legend()




    pred_array = np.array(pred[35:36])[0]
    mu,std = norm.fit(pred_array)
    xmin, xmax = pred_array.min(),pred_array.max()
    x = np.arange(xmin, xmax, 0.01)
    pdf = norm.pdf(x, mu, std)
    plt.plot(x,pdf,label = "normal fit")
    
    
    
    # weights = np.ones_like(pred_array)/float(len(pred_array))
    # plt.hist(pred_array, weights=weights,bins=100)

    plt.plot(x,norm.cdf(x, loc=mu, scale=std),label='cumulative density function')
    plt.xlabel("grade")
    plt.ylabel("cumulative probability")
    plt.legend()

    
    
    
    
    
          
        
        
        
        
        
        