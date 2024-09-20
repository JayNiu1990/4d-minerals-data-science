if __name__ == '__main__':    
    from latent_variable_models_util import n_true, mu_true, sigma_true
    from latent_variable_models_util import generate_data, plot_data, plot_densities
    import daft
    import numpy as np
    import matplotlib.pyplot as plt
    
    from scipy.stats import multivariate_normal as mvn
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np
    from scipy.stats import multivariate_normal as mvn
    
    def m_step(X, q):
        """
        Computes parameters from data and posterior probabilities.
    
        Args:
            X: data (N, D).
            q: posterior probabilities (N, C).
    
        Returns:
            tuple of
            - pi: prior probabilities (C,).
            - mu: mixture component means (C, D).
            - sigma: mixture component covariances (C, D, D).
        """    
        
        N, D = X.shape
        C = q.shape[1]    
        sigma = np.zeros((C, D, D))
        
        # Equation (16)
        pi = np.sum(q, axis=0) / N
    
        # Equation (17)
        mu = q.T.dot(X) / np.sum(q.T, axis=1, keepdims=True)
    
        # Equation (18)
        for c in range(C):
            delta = (X - mu[c])
            sigma[c] = (q[:, [c]] * delta).T.dot(delta) / np.sum(q[:, c])
            
        return pi, mu, sigma    
    
    def lower_bound(X, pi, mu, sigma, q):
        """
        Computes lower bound from data, parameters and posterior probabilities.
    
        Args:
            X: observed data (N, D).
            pi: prior probabilities (C,).
            mu: mixture component means (C, D).
            sigma: mixture component covariances (C, D, D).
            q: posterior probabilities (N, C).
    
        Returns:
            Lower bound.
        """    
    
        N, C = q.shape
        ll = np.zeros((N, C))
    
        # Equation (19)
        for c in range(C):
            ll[:,c] = mvn(mu[c], sigma[c]).logpdf(X)
        return np.sum(q * (ll + np.log(pi) - np.log(np.maximum(q, 1e-8))))
    def random_init_params(X, C):
        D = X.shape[1]
        pi = np.ones(C) / C   ##average prior weight
        ##randomly sample mu from a mvn
        #mu = mvn(mean=np.mean(X, axis=0), cov=[np.var(X[:, 0]), np.var(X[:, 1]), np.var(X[:, 2]), np.var(X[:, 3]), np.var(X[:, 4])]).rvs(C).reshape(C, D)
        mu = mvn(mean=np.mean(X, axis=0), cov=[np.var(X[:, 0])]).rvs(C).reshape(C, D)
        sigma = np.tile(np.eye(D), (C, 1, 1))
        return pi, mu, sigma
    def e_step(X, pi, mu, sigma):
        """
        Computes posterior probabilities from data and parameters.
        
        Args:
            X: observed data (N, D).
            pi: prior probabilities (C,).
            mu: mixture component means (C, D).
            sigma: mixture component covariances (C, D, D).
    
        Returns:
            Posterior probabilities (N, C).
        """
    
        N = X.shape[0]
        C = mu.shape[0]
        q = np.zeros((N, C))
        #print(mvn(mu[0], sigma[0]).pdf(X) * pi[0])
        # Equation (6)
        for c in range(C):
            q[:, c] = mvn(mu[c], sigma[c]).pdf(X) * pi[c]   
    
    
        return q / np.sum(q, axis=-1, keepdims=True) ###sum of q is equal to 1
    def train(X, C, n_restarts=10, max_iter=20, rtol=1e-3):
        q_best = None
        pi_best = None
        mu_best = None
        sigma_best = None
        lb_best = -np.inf
        for _ in range(n_restarts):
            pi, mu, sigma = random_init_params(X, C)
            prev_lb = None
            try:
                for _ in range(max_iter):
                    q = e_step(X, pi, mu, sigma)
                    pi, mu, sigma = m_step(X, q)
    
                    lb = lower_bound(X, pi, mu, sigma, q)
    
                    if lb > lb_best:
                        q_best = q
                        pi_best = pi
                        mu_best = mu
                        sigma_best = sigma
                        lb_best = lb
    
                    if prev_lb and np.abs((lb - prev_lb) / prev_lb) < rtol:
                        break
    
                    prev_lb = lb
            except np.linalg.LinAlgError:
                # Singularity. One of the components collapsed
                # onto a specific data point. Start again ...
                pass
        return pi_best, mu_best, sigma_best, q_best, lb_best
    
    
    # X, T = generate_data(n=n_true, mu=mu_true, sigma=sigma_true)
    # print(X.shape)
    # print(T)
    # plot_data(X, color='grey')
    # plot_data(X, color=T)
    # plot_densities(X, mu=mu_true, sigma=sigma_true,alpha=1)
    
    
    # pi_best, mu_best, sigma_best, q_best, lb_best = train(X, C=3)    
    # print(f'Lower bound = {lb_best:.2f}')
    # print(sum(q_best[0]))
    # plot_data(X, color=q_best)
    # plot_densities(X, mu=mu_best, sigma=sigma_best)
    
    # import matplotlib.pyplot as plt
    # Cs = range(1,8)
    # lbs = []
    # for C in Cs:
    #     lb = train(X,C)[-1]
    #     lbs.append(lb)
    # plt.plot(Cs,lbs)
    # plt.xlabel('Number of mixture components (latent variables)')
    # plt.ylabel('Lower bound')
    # from sklearn.mixture import GaussianMixture
    # gmm = GaussianMixture(n_components=3,n_init=10)
    # gmm.fit(X)
    # plot_data(X, color=gmm.predict_proba(X))
    # plot_densities(X, mu=gmm.means_, sigma=gmm.covariances_)
    
    import pandas as pd
    import warnings
    import matplotlib.pyplot as plt
    import plotly.express as px
    import numpy as np
    import pandas as pd    
    import plotly.io as pio
    import plotly.graph_objs as go
    fields = ['BHID','Fe_dh','As_dh','CuT_dh',"X","Y","Z","LITH","AL_ALT"]
    pio.renderers.default='browser'
    df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)
    df = df.dropna()
    df = df[(pd.to_numeric(df["CuT_dh"], errors='coerce')>0.02) & (pd.to_numeric(df["Fe_dh"], errors='coerce')>0)& (pd.to_numeric(df["As_dh"], errors='coerce')>0)
            & (pd.to_numeric(df["X"], errors='coerce')>=17000)& (pd.to_numeric(df["X"], errors='coerce')<17500)
            & (pd.to_numeric(df["Y"], errors='coerce')>=107000)& (pd.to_numeric(df["Y"], errors='coerce')<107500)
            & (pd.to_numeric(df["Z"], errors='coerce')>=2500)& (pd.to_numeric(df["Z"], errors='coerce')<3000)
            & (pd.to_numeric(df["AL_ALT"], errors='coerce')>=50)]
    
    df['LITH'] = df['LITH'].astype(int)
    df['AL_ALT'] = df['AL_ALT'].astype(int)
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
    import scipy.stats as stats
    df['X_zscores'] = stats.zscore(df['X'])
    df['Y_zscores'] = stats.zscore(df['Y'])
    df['Z_zscores'] = stats.zscore(df['Z'])
    

    #data = np.array(df[['CuT_dh_log_noise']])


    #plt.hist(df['AL_ALT'])
    # import matplotlib.pyplot as plt   
    # Cs = range(2,20)
    # lbs = []
    # for C in Cs:
    #     print(C)
    #     lb = train(data,C)[-1]
    #     lbs.append(lb)
    # plt.plot(Cs,lbs)
    # plt.xlabel('Number of mixture components (latent variables)')
    # plt.ylabel('Lower bound')
    
    # plt.hist(df['latent'])
    # plt.hist(df['LITH'])
    # df.groupby(['latent']).size()
    # plt.hist(df['X_zscores'] ,bins=np.arange(-2,2,0.1))
    #data = np.array(df[['X','Y','Z','CuT_dh','Fe_dh']])
    # x_zscores = stats.zscore(data[:,4])
    # plt.hist(x_zscores,bins=np.arange(0,5,0.1))
    # lith = np.array(df['LITH']).astype('int').reshape(-1, 1)
    #data = np.array(df[['CuT_dh_log_noise','X_zscores','Y_zscores','Z_zscores']])
    n = 50
    xx1 = np.arange(17000, 17500, n).astype('float64')
    yy1 = np.arange(107000, 107500, n).astype('float64')
    zz1 = np.arange(2500, 3000, n).astype('float64')
    
    blocks = []
    for k in zz1:
        for j in yy1:
            for i in xx1:
                sub_block = df.loc[(pd.to_numeric(df["X"], errors='coerce')>=i) & (pd.to_numeric(df["X"], errors='coerce')<i+n) &
                             (pd.to_numeric(df["Y"], errors='coerce')>=j) & (pd.to_numeric(df["Y"], errors='coerce')<j+n)
                             &(pd.to_numeric(df["Z"], errors='coerce')>=k) & (pd.to_numeric(df["Z"], errors='coerce')<k+n)]
                blocks.append(sub_block)
    blocks1 = []
    for i,j in enumerate(blocks):
        if len(j)>=5:
            blocks1.append(j)
    for i, j in enumerate(blocks1):
        blocks1[i]['blocks'] = i
        
    df2_new = pd.concat(blocks1)    
    
    list1 = []
    num=0
    for i in df2_new['AL_ALT'].unique():
        df_sub = df2_new[df2_new['AL_ALT']==i]
        df_sub['alteration'] = num
        if len(df_sub)>10:
            #print(df_sub['alteration'] )
            list1.append(df_sub)
            num+=1
    df = pd.concat(list1)
    
    data = np.array(df[['CuT_dh_log_noise']])
    
    pi_best, mu_best, sigma_best, q_best, lb_best = train(data, C=3)    
    np.argmax(q_best,axis=1)
    
    pred = np.argmax(q_best,axis=1).reshape(-1,1)
    df['latent'] = pred
    