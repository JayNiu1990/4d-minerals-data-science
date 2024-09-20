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
    import random
    from sklearn.metrics import mean_squared_error
    from scipy.stats import invgamma
    fields = ['BHID','Fe_dh','As_dh','CuT_dh',"X","Y","Z","LITH"]
    pio.renderers.default='browser'
    df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)
    df = df.dropna()
    df = df[(pd.to_numeric(df["CuT_dh"], errors='coerce')>0.5) & (pd.to_numeric(df["Fe_dh"], errors='coerce')>0)& (pd.to_numeric(df["As_dh"], errors='coerce')>0)
            & (pd.to_numeric(df["X"], errors='coerce')>=16000)& (pd.to_numeric(df["X"], errors='coerce')<16500)
            & (pd.to_numeric(df["Y"], errors='coerce')>=106500)& (pd.to_numeric(df["Y"], errors='coerce')<107000)
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
    # mu, sigma = 0.1, 0.01
    
    # np.random.seed(0)
    # noise = pd.DataFrame(np.random.normal(mu, sigma, [len(df),1])) 
    
    # noise = round(noise,2)
    # noise.columns = ['noise']
    
    df['CuT_dh_transfered'] = df['CuT_dh'] #stats.zscore(df['CuT_dh'])#df['CuT_dh'] ##stats.zscore(df['CuT_dh'])
    df['CuT_dh_transfered'] = round(df['CuT_dh_transfered'],3)
    
    df['Fe_dh_transfered'] = df['Fe_dh'] #stats.zscore(df['Fe_dh']) #df['Fe_dh'] ##stats.zscore(df['Fe_dh'])
    df['Fe_dh_transfered'] = round(df['Fe_dh_transfered'],3)
    
    df['As_dh_transfered'] = df['As_dh'] #stats.zscore(df['As_dh'])# df['As_dh'] ##stats.zscore(df['As_dh'])
    df['As_dh_transfered'] = round(df['As_dh_transfered'],3)
    
    # df_new1 = pd.concat([df['CuT_dh_log'],noise['noise']],axis=1)
    # df_new1['CuT_dh_log_noise'] = df_new1.sum(axis=1)
    # df = pd.concat([df,df_new1],axis=1)
    
    # df_new2 = pd.concat([df['Fe_dh_log'],noise['noise']],axis=1)
    # df_new2['Fe_dh_log_noise'] = df_new2.sum(axis=1)
    # df = pd.concat([df,df_new2],axis=1)
    
    # df_new3 = pd.concat([df['As_dh_log'],noise['noise']],axis=1)
    # df_new3['As_dh_log_noise'] = df_new3.sum(axis=1)
    # df = pd.concat([df,df_new3],axis=1)
    
    df2 = df[['BHID','X','Y','Z','CuT_dh','Fe_dh','As_dh','CuT_dh_transfered','Fe_dh_transfered','As_dh_transfered','LITH']]
    df2['Cu'] = df2['CuT_dh_transfered']
    df2['Fe'] = df2['Fe_dh_transfered']
    #df2.groupby(['LITH']).size()
    #df2 = df2.loc[df2['LITH']==31]
    #df2 = df2.reset_index(drop=True)
    
    
    n = 100
    m = 50
    xx1 = np.arange(16000, 16500, n).astype('float64')
    yy1 = np.arange(106500, 107000, n).astype('float64')
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
    block_idxs1 = np.array(df2_new['blocks'])
    n_blocks = len(df2_new['blocks'].unique())
    
    # fig = px.scatter_3d(df2_new, x="X",y="Y",z="Z",color="Cu")
    # fig.update_traces(marker_size=3)
    # fig.update_layout(font=dict(size=22))
    # fig.update_layout(scene_aspectmode='data')
    # fig.show()    

    # fig, axis = plt.subplots(1,1,figsize=(12,6))
    # axis.hist(df2_new.groupby(['blocks']).size(),bins=50,color='b')
    # axis.set_xlim(0,230)
    # axis.set_ylim(0,20)
    # axis.set_xlabel('Number of bore core data',fontsize=18)
    # axis.set_ylabel('Frequency',fontsize=18)
    # axis.tick_params(axis='both', which='major', labelsize=18)
    #fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Conference\\AusIMM-2023\\Fig.2.png',dpi=300) 
    
    import pymc3 as pm
    import time
    import warnings
    import matplotlib.pyplot as plt
    import plotly.express as px
    import numpy as np
    import pandas as pd    
    import plotly.io as pio
    import plotly.graph_objs as go
    import arviz as az
    from scipy import linalg, stats
    import time

    def profile_timer(f, *args, **kwargs):
        """
        Times a function call f() and prints how long it took in seconds
        (to the nearest millisecond).
        :param func:  the function f to call
        :return:  same return values as f
        """
        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        print ("time to run {}: {:.3f} sec".format(f.__name__, t1-t0))
        return result
    
    
    class OutlierRegressionMixture(object):
        def __init__(self, y, phi_x, p):
            self.y = y
            self.phi_x = phi_x
            self.p = p

        def log_likelihood(self, theta):
            """
            Mixture likelihood accounting for outliers
            """
            w,v1,v2 = theta[0:2],theta[2:3],theta[3:4]
            resids = self.y - np.dot(w, self.phi_x)
            # Each mixture component is a Gaussian with baseline or inflated variance
            S2_in,S2_out = v1,v2
            exp_in  = np.exp(-0.5*resids**2/S2_in)/np.sqrt(2*np.pi*S2_in)
            exp_out = np.exp(-0.5*resids**2/S2_out)/np.sqrt(2*np.pi*S2_out)
            # The final log likelihood sums over the log likelihoods for each point
            logL = np.sum(np.log((1-self.p)*exp_in + self.p*exp_out))
            return logL
        def log_prior(self, theta):
            """
            Priors over parameters 
            """
            w,v1,v2 = theta[0:2],theta[2:3],theta[3:4]
            # alpha1 = stats.uniform.rvs(1,10)
            # beta1 = stats.uniform.rvs(0,0.2)
            # alpha2 = stats.uniform.rvs(0,1)
            # beta2 = stats.uniform.rvs(0,0.2)
            # DANGER:  improper uniform for now, assume data are good enough
            return 0.0 + np.log(invgamma.pdf(v1, 10, scale = 0.5)) + np.log(invgamma.pdf(v2, 1, scale = 1))
        def log_posterior(self, theta):
            logpost = self.log_prior(theta) + self.log_likelihood(theta)
            if np.isnan(logpost):
                return -np.inf
            return logpost
        def __call__(self, theta):
            return self.log_posterior(theta)

    class MHSampler(object):
        """
        Run a Metropolis-Hastings algorithm given a Model and Proposal.
        """

        def __init__(self, model, proposal, debug=False):
            """
            Initialize a Sampler with a model, a proposal, data, and a guess
            at some reasonable starting parameters.
            :param model: callable accepting a np.array parameter vector
                of shape matching the initial guess theta0, and returning
                a probability (such as a posterior probability)
            :param proposal: callable accepting a np.array parameter vector
                of shape matching the initial guess theta0, and returning
                a proposal of the same shape, as well as the log ratio
                    log (q(theta'|theta)/q(theta|theta'))
            :param theta0: np.array of shape (Npars,)
            :param debug: Boolean flag for whether to turn on the debugging
                print messages in the sample() method
            """
            self.model = model
            self.proposal = proposal
            self._chain_thetas = [ ]
            self._chain_logPs = [ ]
            self._debug = debug

        def run(self, theta0, Nsamples):
            """
            Run the Sampler for Nsamples samples.
            """
            self._chain_thetas = [ theta0 ]
            self._chain_logPs = [ self.model(theta0) ]
            for i in range(Nsamples):
                theta, logpost = self.sample()
                self._chain_thetas.append(theta)
                self._chain_logPs.append(logpost)
            self._chain_thetas = np.array(self._chain_thetas)
            self._chain_logPs = np.array(self._chain_logPs)

        def sample(self):
            """
            Draw a single sample from the MCMC chain, and accept or reject
            using the Metropolis-Hastings criterion.
            """
            theta_old = self._chain_thetas[-1]
            logpost_old = self._chain_logPs[-1]
            theta_prop, logqratio = self.proposal(theta_old)
            if logqratio is -np.inf:
                # flag that this is a Gibbs sampler, auto-accept and skip the rest,
                # assuming the modeler knows what they're doing
                return theta_prop, logpost
            logpost = self.model(theta_prop)
            mhratio = min(1, np.exp(logpost - logpost_old - logqratio))
            if self._debug:
                # this can be useful for sanity checks
                print("theta_old, theta_prop =", theta_old, theta_prop)
                print("logpost_old, logpost_prop =", logpost_old, logpost)
                print("logqratio =", logqratio)
                print("mhratio =", mhratio)
            if np.random.uniform() < mhratio:
                return theta_prop, logpost
            else:
                return theta_old, logpost_old
            
        def chain(self):
            """
            Return a reference to the chain.
            """
            return self._chain_thetas
        
        def accept_frac(self):
            """
            Calculate and return the acceptance fraction.  Works by checking which
            parameter vectors are the same as their predecessors.
            """
            samesame = (self._chain_thetas[1:] == self._chain_thetas[:-1])
            if len(samesame.shape) == 1:
                samesame = samesame.reshape(-1, 1)
            samesame = np.all(samesame, axis=1)
            return 1.0 - (np.sum(samesame) / np.float(len(samesame)))

    # Stub for MCMC stuff

    class GaussianProposal(object):
        """
        A standard isotropic Gaussian proposal for Metropolis Random Walk.
        """
        
        def __init__(self, stepsize):
            """
            :param stepsize:  either float or np.array of shape (d,)
            """
            self.stepsize = stepsize
            
        def __call__(self, theta):
            """
            :param theta:  parameter vector = np.array of shape (d,)
            :return: tuple (logpost, logqratio)
                logpost = log (posterior) density p(y) for the proposed theta
                logqratio = log(q(x,y)/q(y,x)) for asymmetric proposals
            """
            # this proposal is symmetric so the Metropolis q-ratio is 1
            return theta + self.stepsize*(np.random.normal(size=4)),0.0 #


    p=0.2
    Nsamp = 20000
    import matplotlib.pyplot as plt
    from scipy.stats import mode
    # ax.set_xlim([0, 3])
    fig,axis = plt.subplots(2,3,figsize=(22,14),sharey=False,sharex=False);  #32 48 53 56 85 122  131
    axis = axis.ravel()
    Rhat_list= []
    MSE_MLE_nonoutliers = []
    MSE_GMM_nonoutliers = []
    MSE_MLE_outliers = []  
    MSE_GMM_outliers = []
    for i,j in zip([32,48,56,85,122,131],np.arange(0,6)):  # [2,43,60,79,95,145] [32,48,56,85,122,131]
        df3= df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_transfered'])
        X = np.array(df3['CuT_dh_transfered'])
        Y = np.array(df3['Fe_dh_transfered'])
        phi_x = np.vstack([X**0, X**1])
        logpost_outl = OutlierRegressionMixture(Y, phi_x, p)
        sampler =  MHSampler(logpost_outl, GaussianProposal([0.1,0.1,0.1,0.1]))
        chain_array = [ ]
        for n in range(4):
            w_0 = np.random.uniform(0,1,size=2)
            v1_0 = np.array(invgamma.rvs(10,loc=0,scale = 0.5)).reshape(-1)
            v2_0 = np.array(invgamma.rvs(1,loc=0,scale = 1)).reshape(-1)
            theta0 = np.concatenate((w_0,v1_0,v2_0))

            profile_timer(sampler.run, np.array(theta0), Nsamp)
            chain_array.append(sampler.chain()[10001:,:])
            
        chain_array = np.array(chain_array)
        flatchain = chain_array.reshape(-1, chain_array.shape[-1])
        wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))
        func_samples = np.dot(flatchain[:,:2], phi_x)
        post_mu = np.mean(func_samples, axis=0)
        Y_pred = np.dot(wML, phi_x)
        MSE_MLE_nonoutliers.append(mean_squared_error(Y,Y_pred))
        MSE_GMM_nonoutliers.append(mean_squared_error(Y,post_mu))
        percentage_outlier = []
        percentage_nonoutlier = []
        # alpha11 = mode(flatchain[:,2:3])[0][0][0]
        # beta11 = mode(flatchain[:,3:4])[0][0][0]
        # alpha22 = mode(flatchain[:,4:5])[0][0][0]
        # beta22 = mode(flatchain[:,5:6])[0][0][0]
        # alpha11 = flatchain[:,2:3].mean(axis=0)
        # beta11 = flatchain[:,3:4].mean(axis=0)
        v11 = flatchain[:,2:3].mean(axis=0)
        v22 = flatchain[:,3:4].mean(axis=0)
        for x,y in zip(X,Y):
            resids = y - np.dot(flatchain[:,0:2].mean(axis=0), np.vstack([x**0, x**1]))
            S2_in, S2_out = v11,v22
            exp_in  = np.exp(-0.5*resids**2/S2_in)/np.sqrt(2*np.pi*S2_in)
            exp_out = np.exp(-0.5*resids**2/S2_out)/np.sqrt(2*np.pi*S2_out)
            percentage_outlier.append(p*exp_out/((1-p)*exp_in + p*exp_out))
            percentage_nonoutlier.append((1-p)*exp_in/((1-p)*exp_in + p*exp_out))

        idx_badpoint_list = [(idx,data1[0],data2[0]) for idx, (data1,data2) in enumerate(zip(percentage_outlier,percentage_nonoutlier)) if data1[0] > data2[0] ]
        if len(idx_badpoint_list)>0:
            idx_badpoint_list = np.vstack(idx_badpoint_list)
            badpoint_index1 = list(idx_badpoint_list[:,0])
            badpoint_index1 = [int(item) for item in badpoint_index1]
            X_badpoint = X[badpoint_index1]
            Y_badpoint = Y[badpoint_index1]
            
        idx_goodpoint_list = [(idx,data1[0],data2[0]) for idx, (data1,data2) in enumerate(zip(percentage_outlier,percentage_nonoutlier)) if data1[0] < data2[0] ]
        if len(idx_goodpoint_list)>0:
            idx_goodpoint_list = np.vstack(idx_goodpoint_list)
            goodpoint_index1 = list(idx_goodpoint_list[:,0])
            goodpoint_index1 = [int(item) for item in goodpoint_index1]
            X_goodpoint = X[goodpoint_index1]
            Y_goodpoint = Y[goodpoint_index1]
            
        if len(idx_badpoint_list)>0:
            axis[j].plot(X_badpoint, Y_badpoint, ls='None', color='r',marker='o', ms=5, label="Outliers")
            axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Non-outliers")
            axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='b', lw=3, label="MLE")
            flatchain = chain_array.reshape(-1, chain_array.shape[-1])
            func_samples = np.dot(flatchain[:,:2], phi_x)
            post_mu = np.mean(func_samples, axis=0)
            post_sig = np.std(func_samples, axis=0)
            axis[j].plot(X, post_mu, ls='--', lw=3, color='black', label="Posterior Mean of GMM")
            axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GMM' + '\n' + '(95.5% confidence intervals)')
            axis[j].set_title('Block No.' + str(i+1),fontsize=18)
            axis[j].tick_params(axis='both', which='major', labelsize=18)
            axis[j].set_xlabel('Cu grade',fontsize=18)
            axis[j].set_ylabel('Fe grade',fontsize=18)
        else:
            axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Data")
            axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='b', lw=3, label="MLE")
            flatchain = chain_array.reshape(-1, chain_array.shape[-1])
            func_samples = np.dot(flatchain[:,:2], phi_x)
            post_mu = np.mean(func_samples, axis=0)
            post_sig = np.std(func_samples, axis=0)
            axis[j].plot(X, post_mu, ls='--', lw=3, color='black', label="Posterior Mean of GMM")
            axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GMM' + '\n' + '(95.5% confidence intervals)')
            axis[j].set_title('Block No.' + str(i+1),fontsize=18)
            axis[j].tick_params(axis='both', which='major', labelsize=18)
            axis[j].set_xlabel('Cu grade',fontsize=18)
            axis[j].set_ylabel('Fe grade',fontsize=18)
    axis[j].legend(loc='center left',bbox_to_anchor=(1, 0.5),fontsize=18)
    #fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Conference\\AusIMM-2023\\Fig.3.png', bbox_inches='tight',dpi=300)
    fig, ax = plt.subplots(1, 1)
    ax.hist(flatchain[:,2:3],bins=100,color='b')
    ax.set_xlabel('v1')
    ax.set_ylabel('frequency')
    
    fig, ax = plt.subplots(1, 1)
    ax.hist(flatchain[:,3:4],bins=100,color='b')
    ax.set_xlabel('v2')
    ax.set_ylabel('frequency')

    
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(0.01,1,50)
    ax.plot(x, invgamma.pdf(x,10, scale = 0.5),
            'r-', lw=2, alpha=0.6, label='invgamma pdf')
    ax.set_xlabel('v1')
    ax.set_ylabel('frequency')
    
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(0.01,5,50)
    ax.plot(x, invgamma.pdf(x, 1, scale = 1),
            'r-', lw=2, alpha=0.6, label='invgamma pdf')
    ax.set_xlabel('v2')
    ax.set_ylabel('frequency')
    


    # fig, ax = plt.subplots(1, 1)
    # x = np.linspace(0.01,2,50)
    # ax.plot(x, invgamma.pdf(x, 1.35, scale = 0.8),
    #         'r-', lw=2, alpha=0.6, label='invgamma pdf')
    
    # invgamma.median(alpha11,loc=0,scale=beta11)
    p#lt.hist(flatchain[:,2:3],bins=50)
    # fig, ax = plt.subplots(1, 1)
    # x = np.linspace(0.01,5,50)
    # ax.plot(x, invgamma.pdf(x, 2, scale = 10),
    #         'r-', lw=2, alpha=0.6, label='invgamma pdf')
    
    
    # fig,axis = plt.subplots(2,3,figsize=(22,14),sharey=False,sharex=False);  #32 48 53 56 85 122  131
    # axis = axis.ravel()
    # Rhat_list= []
    # MSE_MLE_nonoutliers = []
    # MSE_GMM_nonoutliers = []
    # MSE_MLE_outliers = []
    # MSE_GMM_outliers = []
    # #[32,48,56,85,122,131]
    # for i,j in zip([32,48,56,85,122,131],np.arange(0,6)):
    #     df3= df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_transfered'])
    #     X = np.array(df3['CuT_dh_transfered'])
    #     Y = np.array(df3['Fe_dh_transfered'])
    #     phi_x = np.vstack([X**0, X**1])
    #     logpost_outl = OutlierRegressionMixture(Y, phi_x,p)
    #     sampler =  MHSampler(logpost_outl, GaussianProposal([1,1,1,1]))
    #     chain_array = [ ]
    #     for n in range(4):
    #         theta0 = np.random.uniform(size=4)
    #         profile_timer(sampler.run, np.array(theta0), Nsamp)
    #         #print("chain.mean, chain.std =", sampler.chain().mean(), sampler.chain().std())
    #         #print("acceptance fraction =", sampler.accept_frac())
    #         chain_array.append(sampler.chain()[10001:,:])
    #     chain_array = np.array(chain_array)
    #     flatchain = chain_array.reshape(-1, chain_array.shape[-1])
    #     #traceplots(chain_array, xnames=['b', 'a'])
    #     #rho_k, tau = autocorr(chain_array[1], 1000, plot=False)
    #     #print("chain_array.shape =", chain_array.shape)
    #     #print("chain.mean =", flatchain.mean(axis=0))
    #     #print("chain.std =", flatchain.std(axis=0))
    #     #print("tau.shape =", tau.shape)
    #     #Rhat = gelman_rubin(chain_array)
    #     #print("psrf =", Rhat)
    #     #Rhat_list.append(Rhat)
    #     wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))
    #     func_samples = np.dot(flatchain[:,:2], phi_x)
    #     post_mu = np.mean(func_samples, axis=0)
    #     percentage = []
    #     sigma11 = flatchain[:,2:3].mean(axis=0)
    #     sigma22 = flatchain[:,3:4].mean(axis=0)
    #     for x,y in zip(X,Y):
    #         resids = y - np.dot(flatchain[:,0:2].mean(axis=0), np.vstack([x**0, x**1]))
    #         S2_in, S2_out = sigma11, sigma22
    #         exp_in  = np.exp(-0.5*resids**2/S2_in)/np.sqrt(2*np.pi*S2_in)
    #         exp_out = np.exp(-0.5*resids**2/S2_out)/np.sqrt(2*np.pi*S2_out)
    #         percentage.append(p*exp_out/((1-p)*exp_in + p*exp_out))
    
    #     idx_badpoint_list = [(idx,data[0]) for idx, data in enumerate(percentage) if data >0.5]
    #     if len(idx_badpoint_list)>0:
    #         idx_badpoint_list = np.vstack(idx_badpoint_list)
    #         badpoint_index1 = list(idx_badpoint_list[:,0])
    #         badpoint_index1 = [int(item) for item in badpoint_index1]
    #         X_badpoint = X[badpoint_index1]
    #         Y_badpoint = Y[badpoint_index1]
            
    #     idx_goodpoint_list = [(idx,data[0]) for idx, data in enumerate(percentage) if data <=0.5]
    #     if len(idx_goodpoint_list)>0:
    #         idx_goodpoint_list = np.vstack(idx_goodpoint_list)
    #         goodpoint_index1 = list(idx_goodpoint_list[:,0])
    #         goodpoint_index1 = [int(item) for item in goodpoint_index1]
    #         X_goodpoint = X[goodpoint_index1]
    #         Y_goodpoint = Y[goodpoint_index1]
    #         Y_goodpoint_pred = post_mu[goodpoint_index1]
    #     MSE_MLE_outliers.append(mean_squared_error(Y,post_mu))
    #     MSE_GMM_outliers.append(mean_squared_error(Y_goodpoint,Y_goodpoint_pred))   
    #     if len(idx_badpoint_list)>0:
    #         axis[j].plot(X_badpoint, Y_badpoint, ls='None', color='r',marker='o', ms=5, label="Outliers")
    #         axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Non-outliers")
    #         axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='b', lw=3, label="MLE")
    #         flatchain = chain_array.reshape(-1, chain_array.shape[-1])
    #         func_samples = np.dot(flatchain[:,:2], phi_x)
    #         post_mu = np.mean(func_samples, axis=0)
    #         post_sig = np.std(func_samples, axis=0)
    #         axis[j].plot(X, post_mu, ls='--', lw=3, color='black', label="Posterior Mean of GMM")
    #         axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GMM' + '\n' + '(95.5% confidence intervals)')
    #         axis[j].set_title('Block No.' + str(i+1),fontsize=18)
    #         axis[j].tick_params(axis='both', which='major', labelsize=18)
    #         axis[j].set_xlabel('Cu grade',fontsize=18)
    #         axis[j].set_ylabel('Fe grade',fontsize=18)
    #     else:
    #         axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Data")
    #         axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='b', lw=3, label="MLE")
    #         flatchain = chain_array.reshape(-1, chain_array.shape[-1])
    #         func_samples = np.dot(flatchain[:,:2], phi_x)
    #         post_mu = np.mean(func_samples, axis=0)
    #         post_sig = np.std(func_samples, axis=0)
    #         axis[j].plot(X, post_mu, ls='--', lw=3, color='black', label="Posterior Mean of GMM")
    #         axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GMM' + '\n' + '(95.5% confidence intervals)')
    #         axis[j].set_title('Block No.' + str(i+1),fontsize=18)
    #         axis[j].tick_params(axis='both', which='major', labelsize=18)
    #         axis[j].set_xlabel('Cu grade',fontsize=18)
    #         axis[j].set_ylabel('Fe grade',fontsize=18)
    # axis[j].legend(loc='center left',bbox_to_anchor=(1, 0.5),fontsize=18)
    #fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Conference\\AusIMM-2023\\Fig.4.png', bbox_inches='tight',dpi=300)



    # fig,axis = plt.subplots(2,3,figsize=(24,14),sharey=False,sharex=False);  #32 48 53 56 85 122  131
    # axis = axis.ravel()
    # Rhat_list= []
    # for i,j,m,c in zip([32,32,32,32,32,32],np.arange(0,6),[0.0,0.1,0.2,0.3,0.4,0.5],['y','c','b','r','b','g']):
    #     df3= df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_transfered'])
    #     X = np.array(df3['CuT_dh_transfered'])
    #     Y = np.array(df3['Fe_dh_transfered'])
    #     sigma2 = 1
    #     phi_x = np.vstack([X**0, X**1])
    #     p=m
    #     Nsamp = 20000
    #     logpost_outl = OutlierRegressionMixture(Y, phi_x, sigma2, V, p)
    #     sampler =  MHSampler(logpost_outl, GaussianProposal([1, 1]))
    #     chain_array = [ ]
    #     for n in range(4):
    #         theta0 = np.random.uniform(size=2)
    #         profile_timer(sampler.run, np.array(theta0), Nsamp)
    #         #print("chain.mean, chain.std =", sampler.chain().mean(), sampler.chain().std())
    #         #print("acceptance fraction =", sampler.accept_frac())
    #         chain_array.append(sampler.chain()[10001:,:])
    #     chain_array = np.array(chain_array)
    #     flatchain = chain_array.reshape(-1, chain_array.shape[-1])
    #     #traceplots(chain_array, xnames=['b', 'a'])
    #     rho_k, tau = autocorr(chain_array[1], 1000, plot=False)

    #     Rhat = gelman_rubin(chain_array)
    #     Rhat_list.append(Rhat)
    #     wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))
    #     percentage = []
    #     for x,y in zip(X,Y):
    #         resids = y - np.dot(flatchain.mean(axis=0), np.vstack([x**0, x**1]))
    #         S2_in, S2_out = 1, 1+100
    #         exp_in  = np.exp(-0.5*resids**2/S2_in)/np.sqrt(2*np.pi*S2_in)
    #         exp_out = np.exp(-0.5*resids**2/S2_out)/np.sqrt(2*np.pi*S2_out)
    #         percentage.append(p*exp_out/((1-p)*exp_in + p*exp_out))
    
    #     idx_badpoint_list = [(idx,data[0]) for idx, data in enumerate(percentage) if data >0.5]
    #     if len(idx_badpoint_list)>0:
    #         idx_badpoint_list = np.vstack(idx_badpoint_list)
    #         badpoint_index1 = list(idx_badpoint_list[:,0])
    #         badpoint_index1 = [int(item) for item in badpoint_index1]
    #         X_badpoint = X[badpoint_index1]
    #         Y_badpoint = Y[badpoint_index1]
            
    #     idx_goodpoint_list = [(idx,data[0]) for idx, data in enumerate(percentage) if data <=0.5]
    #     if len(idx_goodpoint_list)>0:
    #         idx_goodpoint_list = np.vstack(idx_goodpoint_list)
    #         goodpoint_index1 = list(idx_goodpoint_list[:,0])
    #         goodpoint_index1 = [int(item) for item in goodpoint_index1]
    #         X_goodpoint = X[goodpoint_index1]
    #         Y_goodpoint = Y[goodpoint_index1]
            
    #     if len(idx_badpoint_list)>0:
    #         axis[j].plot(X_badpoint, Y_badpoint, ls='None', color='r',marker='o', ms=5, label="Outliers")
    #         axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Non-outliers")
    #         axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='m', lw=2, label="MLE")
    #         flatchain = chain_array.reshape(-1, chain_array.shape[-1])
    #         func_samples = np.dot(flatchain[:,:2], phi_x)
    #         post_mu = np.mean(func_samples, axis=0)
    #         post_sig = np.std(func_samples, axis=0)
    #         axis[j].plot(X, post_mu, ls='--', lw=2, color=str(c), label='Posterior Mean of GMM' + '\n' + 'with' + '  ' + str(m) + '  ' + 'p')
    #         #axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GMM' + '\n' + '(95.5% confidence intervals)')
    #         axis[j].set_title('Block No.' + str(i+1),fontsize=18)
    #         axis[j].tick_params(axis='both', which='major', labelsize=18)
    #         axis[j].set_xlabel('Cu grade',fontsize=18)
    #         axis[j].set_ylabel('Fe grade',fontsize=18)
    #         axis[j].legend(loc='upper right',fontsize=16)
    #     else:
    #         axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Data")
    #         axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='m', lw=2, label="MLE")
    #         flatchain = chain_array.reshape(-1, chain_array.shape[-1])
    #         func_samples = np.dot(flatchain[:,:2], phi_x)
    #         post_mu = np.mean(func_samples, axis=0)
    #         post_sig = np.std(func_samples, axis=0)
    #         axis[j].plot(X, post_mu, ls='--', lw=2, color=str(c), label='Posterior Mean of GMM' + '\n' + 'with' + '  ' + str(m) + '  ' + 'p')
    #         #axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GMM' + '\n' + '(95.5% confidence intervals)')
    #         axis[j].set_title('Block No.' + str(i+1),fontsize=18)
    #         axis[j].tick_params(axis='both', which='major', labelsize=18)
    #         axis[j].set_xlabel('Cu grade',fontsize=18)
    #         axis[j].set_ylabel('Fe grade',fontsize=18)
    #         axis[j].legend(loc='upper right',fontsize=16)
    # fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Conference\\AusIMM-2023\\Fig.5.png', bbox_inches='tight',dpi=300)

###########extract all outliers in all blocks
    # p=0.2
    # V=100
    # sigma2 = 1
    # outliers = []
    # Nsamp = 2000
    # Rhat_list= []
    # df2_new=df2_new.reset_index(drop=True)
    # for i in np.arange(0,len(blocks1)):
    #     df3= df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_transfered'])
    #     X = np.array(df3['CuT_dh_transfered'])
    #     Y = np.array(df3['Fe_dh_transfered'])
    #     phi_x = np.vstack([X**0, X**1])

    #     logpost_outl = OutlierRegressionMixture(Y, phi_x, sigma2, V, p)
    #     sampler =  MHSampler(logpost_outl, GaussianProposal([1, 1]))
    #     chain_array = [ ]
    #     for n in range(4):
    #         theta0 = np.random.uniform(size=2)
    #         profile_timer(sampler.run, np.array(theta0), Nsamp)
    #         #print("chain.mean, chain.std =", sampler.chain().mean(), sampler.chain().std())
    #         #print("acceptance fraction =", sampler.accept_frac())
    #         chain_array.append(sampler.chain()[1001:,:])
    #     chain_array = np.array(chain_array)
    #     flatchain = chain_array.reshape(-1, chain_array.shape[-1])

    #     percentage = []
    #     for x,y in zip(X,Y):
    #         resids = y - np.dot(flatchain.mean(axis=0), np.vstack([x**0, x**1]))
    #         S2_in, S2_out = 1, 1+100
    #         exp_in  = np.exp(-0.5*resids**2/S2_in)/np.sqrt(2*np.pi*S2_in)
    #         exp_out = np.exp(-0.5*resids**2/S2_out)/np.sqrt(2*np.pi*S2_out)
    #         percentage.append(p*exp_out/((1-p)*exp_in + p*exp_out))
    #     idx_badpoint_list = [idx for idx, data in enumerate(percentage) if data >0.5]
    #     sub_df3 = df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_transfered'])
    #     outliers.append(sub_df3.iloc[idx_badpoint_list])
    
    # outliers1 = pd.concat(outliers)
    # outliers1 = outliers1.reset_index(drop=True)
    
    # df2_new_nonoutliers = pd.concat([df2_new,outliers1]).drop_duplicates(keep=False)
    # df2_new_nonoutliers['data type'] = 'non-outliers'
    # outliers1['data type'] = 'outliers'
    # df2_new1 = pd.concat([df2_new_nonoutliers,outliers1])
    # fig = px.scatter_3d(outliers1, x="X",y="Y",z="Z",color="data type")
    # fig.update_traces(marker_size=3)
    # fig.update_layout(font=dict(size=22))
    # fig.update_layout(scene_aspectmode='data')
    # fig.show()    
    ###calculate MSE

































