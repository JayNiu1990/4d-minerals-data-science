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
    from sklearn.metrics import mean_squared_error
    from scipy.stats import invgamma
    fields = ['Sample',"Longitude","Latitude","pH","Eh","TDS"]
    pio.renderers.default='browser'
    df = pd.read_excel("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\Continental+Scale+Hydro+Data+Release+April+2023.xlsx", usecols=fields)
    
    
    df1 = df[(pd.to_numeric(df["TDS"], errors='coerce')>0)]
    df1 = df1.dropna(subset=['pH'])
    df1 = df1.dropna(subset=['TDS'])
    df1 = df1.reset_index(drop=True)
    
    df1['pH'] = df1['pH'].astype('float')
    df1['TDS'] = df1['TDS'].astype('float')
    df1['pH'] = np.log10(df1['pH'])
    df1['TDS'] = np.log10(df1['TDS'])
    
    n = 0.5
    m = 0.5
    xx1 = np.arange(round(df1['Longitude'].min(),0)-1, round(df1['Longitude'].max(),0)+1, n).astype('float64')
    yy1 = np.arange(round(df1['Latitude'].min(),0)-1, round(df1['Latitude'].max(),0)+1, n).astype('float64')
    
    blocks = []
    for j in yy1:
        for i in xx1:
            sub_block = df1.loc[(pd.to_numeric(df1["Longitude"], errors='coerce')>=i) & (pd.to_numeric(df1["Longitude"], errors='coerce')<i+n) &
                         (pd.to_numeric(df1["Latitude"], errors='coerce')>=j) & (pd.to_numeric(df1["Latitude"], errors='coerce')<j+n)]
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
    
    df2_new = df2_new.reset_index(drop=True)
    #fig = px.scatter_3d(df1, x="Longitude",y="Latitude",color="pH")
    fig = px.scatter(df1, x="Longitude",y="Latitude",color="pH")
    fig.update_traces(marker_size=3)
    fig.update_layout(font=dict(size=28))
    fig.update_layout(scene_aspectmode='data')
    fig.show()    

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

    def gelman_rubin(data):
        """
        Apply Gelman-Rubin convergence diagnostic to a bunch of chains.
        :param data: np.array of shape (Nchains, Nsamples, Npars)
        """
        Nchains, Nsamples, Npars = data.shape
        B_on_n = data.mean(axis=1).var(axis=0)      # variance of in-chain means
        W = data.var(axis=1).mean(axis=0)           # mean of in-chain variances

        # simple version, as in Obsidian -- not reliable on its own!
        sig2 = (Nsamples/(Nsamples-1))*W + B_on_n
        Vhat = sig2 + B_on_n/Nchains
        Rhat = Vhat/W

        # advanced version that accounts for ndof
        m, n = np.float(Nchains), np.float(Nsamples)
        si2 = data.var(axis=1)
        xi_bar = data.mean(axis=1)
        xi2_bar = data.mean(axis=1)**2
        var_si2 = data.var(axis=1).var(axis=0)
        allmean = data.mean(axis=1).mean(axis=0)
        cov_term1 = np.array([np.cov(si2[:,i], xi2_bar[:,i])[0,1]
                              for i in range(Npars)])
        cov_term2 = np.array([-2*allmean[i]*(np.cov(si2[:,i], xi_bar[:,i])[0,1])
                              for i in range(Npars)])
        var_Vhat = ( ((n-1)/n)**2 * 1.0/m * var_si2
                  +   ((m+1)/m)**2 * 2.0/(m-1) * B_on_n**2
                  +   2.0*(m+1)*(n-1)/(m*n**2)
                        * n/m * (cov_term1 + cov_term2))
        df = 2*Vhat**2 / var_Vhat
        print ("gelman_rubin(): var_Vhat = {}, df = {}".format(var_Vhat, df))
        Rhat *= df/(df-2)
        
        return Rhat
    def autocorr(x, D, plot=True):
        """
        Discrete autocorrelation function + integrated autocorrelation time.
        Calculates directly, though could be sped up using Fourier transforms.
        See Daniel Foreman-Mackey's tutorial (based on notes from Alan Sokal):
        https://emcee.readthedocs.io/en/stable/tutorials/autocorr/

        :param x: np.array of data, of shape (Nsamples, Ndim)
        :param D: number of return arrays
        """
        # Baseline discrete autocorrelation:  whiten the data and calculate
        # the mean sample correlation in each window
        xp = np.atleast_2d(x)
        z = (xp-np.mean(xp, axis=0))/np.std(xp, axis=0)
        Ct = np.ones((D, z.shape[1]))
        Ct[1:,:] = np.array([np.mean(z[i:]*z[:-i], axis=0) for i in range(1,D)])
        # Integrated autocorrelation tau_hat as a function of cutoff window M
        tau_hat = 1 + 2*np.cumsum(Ct, axis=0)
        # Sokal's advice is to take the autocorrelation time calculated using
        # the smallest integration limit M that's less than 5*tau_hat[M]
        Mrange = np.arange(len(tau_hat))
        tau = np.argmin(Mrange[:,None] - 5*tau_hat, axis=0)
        print("tau =", tau)
        # Plot if requested
        if plot:
            fig = plt.figure(figsize=(6,4))
            plt.plot(Ct)
            plt.title('Discrete Autocorrelation ($\\tau = {:.1f}$)'.format(np.mean(tau)))
        return np.array(Ct), tau
    def traceplots(x, xnames=None, title=None):
        """
        Runs trace plots.
        :param x:  np.array of shape (N, d)
        :param xnames:  optional iterable of length d, containing the names
            of variables making up the dimensions of x (used as y-axis labels)
        :param title:  optional plot title
        """
        # set out limits of plot spaces, in dimensionless viewport coordinates
        # that run from 0 (bottom, left) to 1 (top, right) along both axes
        D, N, d = x.shape
        fig, axis = plt.subplots(2,1,figsize=(12,8))
        axis = axis.ravel()
        left, tracewidth, histwidth = 0.1, 0.65, 0.15
        bottom, rowheight = 0.1, 0.8/d
        spacing = 0.05
        for j in range(D):
            for i in range(d):
                axis[i].plot(x[j,:,i])
        axis[0].set_ylabel('b',fontsize=18)
        axis[1].set_ylabel('a',fontsize=18)
        axis[1].set_xlabel('Sample index',fontsize=16)
        axis[0].tick_params(axis='both', which='major', labelsize=14)
        axis[1].tick_params(axis='both', which='major', labelsize=14)

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
        
        def __init__(self, y, phi_x, sigma2, V, p):
            self.y = y
            self.phi_x = phi_x
            self.sigma2 = sigma2
            self.V = V
            self.p = p

            
        def log_likelihood(self, theta):
            """
            Mixture likelihood accounting for outliers
            """
            # Form regression mean and residuals
            w = theta
            resids = self.y - np.dot(w, self.phi_x)
            # Each mixture component is a Gaussian with baseline or inflated variance
            S2_in, S2_out = self.sigma2, self.sigma2 + self.V
            exp_in  = np.exp(-0.5*resids**2/S2_in)/np.sqrt(2*np.pi*S2_in)
            exp_out = np.exp(-0.5*resids**2/S2_out)/np.sqrt(2*np.pi*S2_out)
            # The final log likelihood sums over the log likelihoods for each point
            logL = np.sum(np.log((1-self.p)*exp_in + self.p*exp_out))
            return logL

        def log_prior(self, theta):
            """
            Priors over parameters 
            """
            # DANGER:  improper uniform for now, assume data are good enough
            return 0.0
            
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

            return theta + self.stepsize*np.random.normal(size=theta.shape), 0.0


    
    import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1)
    # a=4
    # x = np.linspace(0.01,10,500)
    # ax.plot(x, invgamma.pdf(x, a),
    #        'r-', lw=2, alpha=0.6, label='invgamma pdf')
    # ax.set_xlim([0, 3])
    # invgamma.rvs(a)

    p=0.2
    V=1
    sigma2 = 0.1
    Nsamp = 2000
    fig,axis = plt.subplots(2,3,figsize=(22,14),sharey=False,sharex=False);  #32 48 53 56 85 122  131
    axis = axis.ravel()
    Rhat_list= []
    MSE_MLE_outliers1 = []
    MSE_GMM_outliers1 = []
    for i,j in zip([12,13,14,15,16,17],np.arange(0,6)):
        df3= df2_new[df2_new['blocks']==i].sort_values(by=['pH'])
        X = np.array(df3['pH'])
        Y = np.array(df3['TDS'])
        phi_x = np.vstack([X**0, X**1])
        logpost_outl = OutlierRegressionMixture(Y, phi_x, sigma2, V, p)
        sampler =  MHSampler(logpost_outl, GaussianProposal([1, 1]))
        chain_array = [ ]
        for n in range(4):
            theta0 = np.random.uniform(size=2)
            profile_timer(sampler.run, np.array(theta0), Nsamp)
            #print("chain.mean, chain.std =", sampler.chain().mean(), sampler.chain().std())
            #print("acceptance fraction =", sampler.accept_frac())
            chain_array.append(sampler.chain()[1001:,:])
        chain_array = np.array(chain_array)
        flatchain = chain_array.reshape(-1, chain_array.shape[-1])
        #traceplots(chain_array, xnames=['b', 'a'])
        # rho_k, tau = autocorr(chain_array[1], 1000, plot=False)
        #print("chain_array.shape =", chain_array.shape)
        #print("chain.mean =", flatchain.mean(axis=0))
        #print("chain.std =", flatchain.std(axis=0))
        #print("tau.shape =", tau.shape)
        # Rhat = gelman_rubin(chain_array)
        #print("psrf =", Rhat)
        # Rhat_list.append(Rhat)
        wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))
        func_samples = np.dot(flatchain[:,:2], phi_x)
        post_mu = np.mean(func_samples, axis=0)
        Y_pred = np.dot(wML, phi_x)

        percentage = []
        for x,y in zip(X,Y):
            resids = y - np.dot(flatchain.mean(axis=0), np.vstack([x**0, x**1]))
            S2_in, S2_out = sigma2, sigma2+V
            exp_in  = np.exp(-0.5*resids**2/S2_in)/np.sqrt(2*np.pi*S2_in)
            exp_out = np.exp(-0.5*resids**2/S2_out)/np.sqrt(2*np.pi*S2_out)
            percentage.append(p*exp_out/((1-p)*exp_in + p*exp_out))

        idx_badpoint_list = [(idx,data[0]) for idx, data in enumerate(percentage) if data >0.5]
        if len(idx_badpoint_list)>0:
            idx_badpoint_list = np.vstack(idx_badpoint_list)
            badpoint_index1 = list(idx_badpoint_list[:,0])
            badpoint_index1 = [int(item) for item in badpoint_index1]
            X_badpoint = X[badpoint_index1]
            Y_badpoint = Y[badpoint_index1]
            
        idx_goodpoint_list = [(idx,data[0]) for idx, data in enumerate(percentage) if data <=0.5]
        if len(idx_goodpoint_list)>0:
            idx_goodpoint_list = np.vstack(idx_goodpoint_list)
            goodpoint_index1 = list(idx_goodpoint_list[:,0])
            goodpoint_index1 = [int(item) for item in goodpoint_index1]
            X_goodpoint = X[goodpoint_index1]
            Y_goodpoint = Y[goodpoint_index1]
            Y_goodpoint_pred = post_mu[goodpoint_index1]
        MSE_MLE_outliers1.append(mean_squared_error(Y,post_mu))
        MSE_GMM_outliers1.append(mean_squared_error(Y_goodpoint,Y_goodpoint_pred))   
        if len(idx_badpoint_list)>0:
            axis[j].plot(X_badpoint, Y_badpoint, ls='None', color='r',marker='o', ms=5, label="Outliers")
            axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Non-outliers")
            axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='b', lw=3, label="MLE")
            flatchain = chain_array.reshape(-1, chain_array.shape[-1])
            func_samples = np.dot(flatchain[:,:2], phi_x)
            post_mu = np.mean(func_samples, axis=0)
            post_sig = np.std(func_samples, axis=0)
            axis[j].plot(X, post_mu, ls='--', lw=3, color='black', label="Posterior Mean of GMM")
            axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GML' + '\n' + '(95.5% confidence intervals)')
            axis[j].set_title('Block No.' + str(i+1),fontsize=18)
            axis[j].tick_params(axis='both', which='major', labelsize=18)
            axis[j].set_xlabel('PH',fontsize=18)
            axis[j].set_ylabel('TDS',fontsize=18)
        else:
            axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Non-outliers")
            axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='b', lw=3, label="MLE")
            flatchain = chain_array.reshape(-1, chain_array.shape[-1])
            func_samples = np.dot(flatchain[:,:2], phi_x)
            post_mu = np.mean(func_samples, axis=0)
            post_sig = np.std(func_samples, axis=0)
            axis[j].plot(X, post_mu, ls='--', lw=3, color='black', label="Posterior Mean of GMM")
            axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GML' + '\n' + '(95.5% confidence intervals)')
            axis[j].set_title('Block No.' + str(i+1),fontsize=18)
            axis[j].tick_params(axis='both', which='major', labelsize=18)
            axis[j].set_xlabel('PH',fontsize=18)
            axis[j].set_ylabel('TDS',fontsize=18)
    axis[j].legend(loc='center left',bbox_to_anchor=(1, 0.5),fontsize=18)
    # fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Conference\\AusIMM-2023\\Fig.3.png', bbox_inches='tight',dpi=300)
    
    plt.hist(flatchain[:,1],bins=50)
    # fig,axis = plt.subplots(2,3,figsize=(22,14),sharey=False,sharex=False);  #32 48 53 56 85 122  131
    # axis = axis.ravel()
    # Rhat_list= []
    # MSE_MLE_outliers2 = []
    # MSE_GMM_outliers2 = []
    # for i,j in zip([32,48,56,85,122,131],np.arange(0,6)):
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
    #         chain_array.append(sampler.chain()[10001:,:])
    #     chain_array = np.array(chain_array)
    #     flatchain = chain_array.reshape(-1, chain_array.shape[-1])
    #     #traceplots(chain_array, xnames=['b', 'a'])
    #     rho_k, tau = autocorr(chain_array[1], 1000, plot=False)
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
    #     for x,y in zip(X,Y):
    #         resids = y - np.dot(flatchain.mean(axis=0), np.vstack([x**0, x**1]))
    #         S2_in, S2_out = sigma2, sigma2+V
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
    #     MSE_MLE_outliers2.append(mean_squared_error(Y,post_mu))
    #     MSE_GMM_outliers2.append(mean_squared_error(Y_goodpoint,Y_goodpoint_pred))   
    #     if len(idx_badpoint_list)>0:
    #         axis[j].plot(X_badpoint, Y_badpoint, ls='None', color='r',marker='o', ms=5, label="Outliers")
    #         axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Non-outliers")
    #         axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='b', lw=3, label="MLE")
    #         flatchain = chain_array.reshape(-1, chain_array.shape[-1])
    #         func_samples = np.dot(flatchain[:,:2], phi_x)
    #         post_mu = np.mean(func_samples, axis=0)
    #         post_sig = np.std(func_samples, axis=0)
    #         axis[j].plot(X, post_mu, ls='--', lw=3, color='black', label="Posterior Mean of GMM")
    #         axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GML' + '\n' + '(95.5% confidence intervals)')
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
    #         axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GML' + '\n' + '(95.5% confidence intervals)')
    #         axis[j].set_title('Block No.' + str(i+1),fontsize=18)
    #         axis[j].tick_params(axis='both', which='major', labelsize=18)
    #         axis[j].set_xlabel('Cu grade',fontsize=18)
    #         axis[j].set_ylabel('Fe grade',fontsize=18)
    # axis[j].legend(loc='center left',bbox_to_anchor=(1, 0.5),fontsize=18)
    # fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Conference\\AusIMM-2023\\Fig.4.png', bbox_inches='tight',dpi=300)



    # fig,axis = plt.subplots(2,3,figsize=(24,14),sharey=False,sharex=False);  #32 48 53 56 85 122  131
    # axis = axis.ravel()
    # Rhat_list= []
    # for i,j,c,sigma2,V in zip([122,122,122,122,122,122],np.arange(0,6),['mediumorchid','c','b','r','b','g'],[0.02,0.05,0.1,0.2,0.3,0.5],[0.5,1,2,5,10,20]):
    #     df3= df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_transfered'])
    #     X = np.array(df3['CuT_dh_transfered'])
    #     Y = np.array(df3['Fe_dh_transfered'])
    #     phi_x = np.vstack([X**0, X**1])
    #     p=0.2
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
    #         S2_in, S2_out = sigma2, sigma2+V
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
    #     if j <1:   
    #         flatchain = chain_array.reshape(-1, chain_array.shape[-1])
    #         func_samples = np.dot(flatchain[:,:2], phi_x)
    #         post_mu = np.mean(func_samples, axis=0)
    #         post_sig = np.std(func_samples, axis=0)
    #         axis[j].set_title('Block No.' + str(i+1),fontsize=18)
    #         axis[j].tick_params(axis='both', which='major', labelsize=18)
    #         axis[j].set_xlabel('Cu grade',fontsize=18)
    #         axis[j].set_ylabel('Fe grade',fontsize=18)
    #         axis[j].plot(X_badpoint, Y_badpoint, ls='None', color='r',marker='o', ms=5, label="Outliers")
    #         axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5, label="Non-outliers")
    #         axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='m', lw=2, label="MLE")
    #         axis[j].plot(X, post_mu, ls='--', lw=2, color=str(c), label='Posterior Mean of GML' + '\n' + 'with sigma='+ str(sigma2) +',V=' + str(V))

    #     else:
    #         if len(idx_badpoint_list)>0:
    #             flatchain = chain_array.reshape(-1, chain_array.shape[-1])
    #             func_samples = np.dot(flatchain[:,:2], phi_x)
    #             post_mu = np.mean(func_samples, axis=0)
    #             post_sig = np.std(func_samples, axis=0)
    #             axis[j].plot(X_badpoint, Y_badpoint, ls='None', color='r',marker='o', ms=5)
    #             axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5)
    #             axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='m', lw=2)
    #             axis[j].plot(X, post_mu, ls='--', lw=2, color=str(c), label='Posterior Mean of GML' + '\n' + 'with sigma='+ str(sigma2) +',V=' + str(V))
    #             #axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GMM' + '\n' + '(95.5% confidence intervals)')
    #             axis[j].set_title('Block No.' + str(i+1),fontsize=18)
    #             axis[j].tick_params(axis='both', which='major', labelsize=18)
    #             axis[j].set_xlabel('Cu grade',fontsize=18)
    #             axis[j].set_ylabel('Fe grade',fontsize=18)
    #             #axis[j].legend(loc='upper right',fontsize=16)
    #         else:
    #             axis[j].plot(X_goodpoint, Y_goodpoint, ls='None', color='black',marker='o', ms=5)
    #             axis[j].plot(X, np.dot(wML, phi_x), ls='--',color='m', lw=2)
    #             flatchain = chain_array.reshape(-1, chain_array.shape[-1])
    #             func_samples = np.dot(flatchain[:,:2], phi_x)
    #             post_mu = np.mean(func_samples, axis=0)
    #             post_sig = np.std(func_samples, axis=0)
    #             axis[j].plot(X, post_mu, ls='--', lw=2, color=str(c), label='Posterior Mean of GML' + '\n' + 'with sigma='+ str(sigma2) +',V=' + str(V))
    #             #axis[j].fill_between(X, post_mu-2*post_sig, post_mu+2*post_sig, color='dodgerblue', alpha=0.3, label='Posterior Variance of GMM' + '\n' + '(95.5% confidence intervals)')
    #             axis[j].set_title('Block No.' + str(i+1),fontsize=18)
    #             axis[j].tick_params(axis='both', which='major', labelsize=18)
    #             axis[j].set_xlabel('Cu grade',fontsize=18)
    #             axis[j].set_ylabel('Fe grade',fontsize=18)
    # fig.legend(loc='center left',bbox_to_anchor=(0.9, 0.5),fontsize=18)
    # fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Conference\\AusIMM-2023\\Fig.5.png', bbox_inches='tight',dpi=300)

###########extract all outliers in all blocks
    # p=0.2
    # V=10
    # sigma2 = 0.1
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
    #         S2_in, S2_out = sigma2, sigma2+V
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

































