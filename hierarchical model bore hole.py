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
    
    df['CuT_dh_log'] = stats.zscore(df['CuT_dh'])#stats.zscore(df['CuT_dh'])
    df['CuT_dh_log'] = round(df['CuT_dh_log'],3)
    
    df['Fe_dh_log'] = stats.zscore(df['Fe_dh'])#stats.zscore(df['Fe_dh'])
    df['Fe_dh_log'] = round(df['Fe_dh_log'],3)
    
    df['As_dh_log'] = stats.zscore(df['As_dh'])#stats.zscore(df['As_dh'])
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
    m = 100
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

    block_idxs1 = np.array(df2_new['blocks'])
    n_blocks = len(df2_new['blocks'].unique())
    import pymc3 as pm
    import time
    
    with pm.Model() as unpooled_model_HalfCauchy:
        eps = pm.HalfCauchy('sigma', 1)
        alpha = pm.Normal('alpha',mu = 0,sd = 1,shape = n_blocks)
        beta = pm.Normal('beta',mu = 0,sd = 1, shape = n_blocks)
        
        Fe_mean = alpha[block_idxs1] + beta[block_idxs1]*df2_new['CuT_dh_log_noise'].values
        Fe = pm.Normal('Fe', mu = Fe_mean, sd = eps,observed = df2_new['Fe_dh_log_noise'])
    with unpooled_model_HalfCauchy:
        unpooled_trace_HalfCauchy = pm.sample(2000)   
    
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
        N, d = x.shape
        fig = plt.figure()
        left, tracewidth, histwidth = 0.1, 0.65, 0.15
        bottom, rowheight = 0.1, 0.8/d
        spacing = 0.05
        
        for i in range(d):
            # Set the location of the trace and histogram viewports,
            # starting with the first dimension from the bottom of the canvas
            rowbottom = bottom + i*rowheight
            rect_trace = (left, rowbottom, tracewidth, rowheight)
            rect_hist = (left + tracewidth, rowbottom, histwidth, rowheight)
            # First set of trace plot axes
            if i == 0:
                ax_trace = fig.add_axes(rect_trace)
                ax_trace.plot(x[:,i])
                ax_trace.set_xlabel("Sample Count")
                ax_tr0 = ax_trace
            # Other sets of trace plot axes that share the first trace's x-axis
            # Make tick labels invisible so they don't clutter up the plot
            elif i > 0:
                ax_trace = fig.add_axes(rect_trace, sharex=ax_tr0)
                ax_trace.plot(x[:,i])
                plt.setp(ax_trace.get_xticklabels(), visible=False)
            # Title at the top
            if i == d-1 and title is not None:
                plt.title(title)
            # Trace y-axis labels
            if xnames is not None:
                ax_trace.set_ylabel(xnames[i])
            # Trace histograms at the right
            ax_hist = fig.add_axes(rect_hist, sharey=ax_trace)
            ax_hist.hist(x[:,i], orientation='horizontal', bins=50)
            #plt.setp(ax_hist.get_xticklabels(), visible=False)
            #plt.setp(ax_hist.get_yticklabels(), visible=False)
            xlim = ax_hist.get_xlim()
            ax_hist.set_xlim([xlim[0], 1.1*xlim[1]])
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

    class OutlierGibbsProposal(object):
        """
        A Gibbs sampling proposal to sample from this model.
        """
        
        def __init__(self, model):
            # Add access to the model just in case we need it
            self.y = model.y
            self.phi_x = model.phi_x
            self.sigma2 = model.sigma2
            self.V = model.V
            self.p = model.p
            self.Nw, self.Nq = model.phi_x.shape
            # Some pre-computed constants
            self.logP_q0_norm = -0.5*np.log(2*np.pi*self.sigma2) + np.log(1-self.p)
            self.logP_q1_norm = -0.5*np.log(2*np.pi*(self.sigma2 + self.V)) + np.log(self.p)
            
        def __call__(self, theta):
            """
            :param theta:  parameter vector = np.array of shape (d,)
            :return: tuple (logpost, logqratio)
                logpost = log (posterior) density p(y) for the proposed theta
                logqratio = log(q(x,y)/q(y,x)) for asymmetric proposals
            """
            w, q = theta[:self.Nw], theta[self.Nw:]

            # Step 1:  propose from P(q|w)
            # Each mixture component has a Gaussian variance which may be inflated
            # The final log likelihood sums over the log likelihoods for each point
            # Conditioned on the w's, figure out the density ratios for q
            resids = self.y - np.dot(w, self.phi_x)
            logP_q0 = -0.5*(resids**2/self.sigma2) + self.logP_q0_norm
            logP_q1 = -0.5*(resids**2/(self.sigma2 + self.V)) + self.logP_q1_norm
            logP_q = np.exp(logP_q1 - logP_q0)
            # Now propose a random set of q's based on those probabilities
            q_prop = (np.random.uniform(q.shape) < logP_q)
            
            # Step 2:  propose from P(w|q)
            # Conditioned on the q's, draw from the conditional density for the w's
            # Helpful that the covariance matrix is diagonal, thus easily inverted
            yvar = self.sigma2 + q_prop*self.V
            Cinv = np.diag(1.0/yvar)
            wprec = np.dot(self.phi_x, np.dot(Cinv, self.phi_x.T))
            Lprec = linalg.cholesky(wprec)
            # The conditional posterior for the weights is multi-Gaussian with
            # mu = (phi*C.inv*phi.T).inv * phi*C.inv*y, var = (phi*C.inv*phi.T).inv
            # More numerically stable to use linalg.solve than linalg.inv
            u = np.random.normal(size=w.shape)
            w_prop = linalg.solve_triangular(Lprec, u)
            w_prop += linalg.solve(wprec, np.dot(self.phi_x, self.y/yvar))

            # Step 3:  Profit!
            # This is a Gibbs step so I'm just going to return -np.inf for the
            # log proposal density ratio, which should make MHSampler auto-accept
            return np.concatenate([w_prop, q_prop]), -np.inf
     
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
    # Stub for MCMC stuff

    class OutlierRegressionLatent(object):
        
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
            w, q = theta
            resids = self.y - np.dot(w, self.phi_x)
            # Each mixture component has a Gaussian variance which may be inflated
            # The final log likelihood sums over the log likelihoods for each point
            S2 = self.sigma2 + q*self.V
            logL = -0.5*np.sum(resids**2/S2 + np.log(2*np.pi*S2))
            return logL

        def log_prior(self, theta):
            """
            Priors over parameters
            """
            # Bernoulli prior for the latents; leave improper uniforms over the weights
            # (don't do this at home, folks, we're just in a rush today)
            w, q = theta
            N, Nout = len(q), np.sum(q)
            return Nout*np.log(p) + (N-Nout)*np.log(1-p)
            
        def log_posterior(self, theta):
            logpost = self.log_prior(theta) + self.log_likelihood(theta)
            if np.isnan(logpost):
                return -np.inf
            return logpost
        
        def conditional_draw(self, theta, i):
            """
            A stub for Gibbs sampling
            """
            pass

        def __call__(self, theta):
            return self.log_posterior(theta)


    from scipy import linalg, stats
    selection2 = [6,7,8,9,10,11]  #list(df2_new.blocks.unique())[493:499] #0-3 #246-249 #493-496
    mean_intercept_list = []
    mean_slope_list = []
    for i in selection2:
        df3= df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_log_noise'])
        #df3= df2[300:600]#.sort_values(by=['CuT_dh'])[0:300]
        X = np.array(df3['CuT_dh_log_noise'])
        Y = np.array(df3['Fe_dh_log_noise'])
        sigma2 = 1
        phi_x = np.vstack([X**0, X**1])
        p=0.2
        V=100.0
        Nsamp = 2000
        logpost_outl = OutlierRegressionLatent(Y, phi_x, sigma2, V, p)
        sampler = MHSampler(lambda theta: np.inf, OutlierGibbsProposal(logpost_outl))
        chain_array = [ ]
        for i in range(4):
            theta0 = np.random.uniform(size=np.sum(phi_x.shape))
            profile_timer(sampler.run, np.array(theta0), Nsamp)
            print("chain.mean, chain.std =", sampler.chain().mean(), sampler.chain().std())
            print("acceptance fraction =", sampler.accept_frac())
            chain_array.append(sampler.chain())
        chain_array = np.array(chain_array)
        flatchain = chain_array.reshape(-1, chain_array.shape[-1])
        traceplots(chain_array[1], #  xnames=['w0', 'w1', 'w2'],
                    title="Outlier Regression Weight Traces")
        rho_k, tau = autocorr(chain_array[1], 1000, plot=False)
        print("chain_array.shape =", chain_array.shape)
        print("chain.mean =", flatchain.mean(axis=0))
        print("chain.std =", flatchain.std(axis=0))
        print("tau.shape =", tau.shape)
        Rhat = gelman_rubin(chain_array)
        print("psrf =", Rhat)
        wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))
        flatchain = chain_array.reshape(-1, chain_array.shape[-1])
        func_samples = np.dot(flatchain[:,:2], phi_x)
        mean_intercept = flatchain[:,:1].reshape(-1).mean()
        mean_slope = flatchain[:,1:2].reshape(-1).mean()
        mean_intercept_list.append(mean_intercept)
        mean_slope_list.append(mean_slope)
        
        
        
        
    fig,axis = plt.subplots(2,3,figsize=(18,8),sharey=False,sharex=False);
    axis = axis.ravel()
    selection2 = [6,7,8,9,10,11]  
    for i,c in enumerate(selection2):
        c_data = df2_new.loc[df2_new.blocks==c]
        c_data = c_data.reset_index(drop=True)
        xvals = np.linspace(-5,2)
        axis[i].plot(xvals, mean_intercept_list[i] + mean_slope_list[i]*xvals,lw=3, color='black', label="Posterior Mean (latent)")
        axis[i].set_title('Block No.' + str(c+1),fontsize=14)
        axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=200,label = 'bore core')
        axis[i].plot(xvals,hirearchical_trace_normal_HalfCauchy['alpha'][:,c].mean()+hirearchical_trace_normal_HalfCauchy['beta'][:,c].mean()*xvals,'c',alpha=1,lw=3.,label='partial pooling')
        axis[i].plot(xvals,unpooled_trace_HalfCauchy['alpha'][:,c].mean()+unpooled_trace_HalfCauchy['beta'][:,c].mean()*xvals,'r',alpha=1,lw=3.,label='no pooling')
        axis[i].set_ylabel('log Fe',fontsize=14)
        # axis[i].set_xlim(-5,5)
        axis[i].set_ylim(-2,5)
        axis[i].tick_params(axis='both', which='major', labelsize=14)
        axis[i].set_xlabel('log Cu',fontsize=14)
        if i ==5:
            axis[i].legend(loc='upper left',prop={'size':12})
    
    
    
    
    X = np.array(df2_new[df2_new['blocks']==8].sort_values(by=['CuT_dh_log_noise'])['CuT_dh_log_noise'])
    Y = np.array(df2_new[df2_new['blocks']==8].sort_values(by=['CuT_dh_log_noise'])['Fe_dh_log_noise'])
    sigma2 = 1
    phi_x = np.vstack([X**0, X**1])
    p=0.1
    V=100.0
    Nsamp = 2000
    logpost_outl = OutlierRegressionLatent(Y, phi_x, sigma2, V, p)
    sampler = MHSampler(lambda theta: np.inf, OutlierGibbsProposal(logpost_outl))
    chain_array = [ ]
    for i in range(5):
        theta0 = np.random.uniform(size=np.sum(phi_x.shape))
        profile_timer(sampler.run, np.array(theta0), Nsamp)
        print("chain.mean, chain.std =", sampler.chain().mean(), sampler.chain().std())
        print("acceptance fraction =", sampler.accept_frac())
        chain_array.append(sampler.chain())
    chain_array = np.array(chain_array)
    flatchain = chain_array.reshape(-1, chain_array.shape[-1])
    traceplots(chain_array[1], #  xnames=['w0', 'w1', 'w2'],
                title="Outlier Regression Weight Traces")
    rho_k, tau = autocorr(chain_array[1], 1000, plot=False)
    print("chain_array.shape =", chain_array.shape)
    print("chain.mean =", flatchain.mean(axis=0))
    print("chain.std =", flatchain.std(axis=0))
    print("tau.shape =", tau.shape)
    Rhat = gelman_rubin(chain_array)
    print("psrf =", Rhat)
    wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))
    # Visualize our answer!
    plt.figure(figsize=(6,4))
    plt.plot(X, Y, ls='None', marker='o', ms=3, label="Data")
    plt.plot(X, np.dot(wML, phi_x), ls='--', lw=2, label="Maximum Likelihood")
    flatchain = chain_array.reshape(-1, chain_array.shape[-1])
    func_samples = np.dot(flatchain[:,:2], phi_x)
    post_mu = np.mean(func_samples, axis=0)
    post_sig = np.std(func_samples, axis=0)
    plt.plot(X, post_mu, ls='--', lw=2, color='dodgerblue', label="Posterior Mean (Latent variable model)")
    plt.fill_between(X, post_mu-post_sig, post_mu+post_sig, color='dodgerblue', alpha=0.5, label="Posterior Variance")
    plt.legend(loc='best')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Regression with Outliers (Latent)")




    fig,axis = plt.subplots(2,3,figsize=(20,12),sharey=True,sharex=False);
    axis = axis.ravel()
    for i,j in zip(np.arange(6,12),np.arange(0,6)):
        X = np.array(df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_log_noise'])['CuT_dh_log_noise'])
        Y = np.array(df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_log_noise'])['Fe_dh_log_noise'])
        sigma2 = 1
        phi_x = np.vstack([X**0, X**1])
        p=0.2
        V=100.0
        Nsamp = 2000
        logpost_outl = OutlierRegressionLatent(Y, phi_x, sigma2, V, p)
        sampler = MHSampler(lambda theta: np.inf, OutlierGibbsProposal(logpost_outl))
        chain_array = [ ]
        for i in range(5):
            theta0 = np.random.uniform(size=np.sum(phi_x.shape))
            profile_timer(sampler.run, np.array(theta0), Nsamp)
            print("chain.mean, chain.std =", sampler.chain().mean(), sampler.chain().std())
            print("acceptance fraction =", sampler.accept_frac())
            chain_array.append(sampler.chain())
        chain_array = np.array(chain_array)
        flatchain = chain_array.reshape(-1, chain_array.shape[-1])
        #traceplots(chain_array[1], #  xnames=['w0', 'w1', 'w2'],
                   # title="Outlier Regression Weight Traces")
        rho_k, tau = autocorr(chain_array[1], 1000, plot=False)
        print("chain_array.shape =", chain_array.shape)
        print("chain.mean =", flatchain.mean(axis=0))
        print("chain.std =", flatchain.std(axis=0))
        print("tau.shape =", tau.shape)
        Rhat = gelman_rubin(chain_array)
        print("psrf =", Rhat)
        wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))
        # Visualize our answer!
        #plt.figure(figsize=(6,4))
        axis[j].plot(X, Y, ls='None', marker='o', ms=3, label="Data")
        axis[j].plot(X, np.dot(wML, phi_x), ls='--', lw=2, label="Maximum Likelihood")
        flatchain = chain_array.reshape(-1, chain_array.shape[-1])
        func_samples = np.dot(flatchain[:,:2], phi_x)
        post_mu = np.mean(func_samples, axis=0)
        post_sig = np.std(func_samples, axis=0)
        axis[j].plot(X, post_mu, ls='--', lw=2, color='dodgerblue', label="Posterior Mean (latent class)")
        axis[j].fill_between(X, post_mu-post_sig, post_mu+post_sig, color='dodgerblue', alpha=0.5, label="Posterior Variance")
    axis[j].legend(loc='best')

    
    selection3 = [6,7,8]  #list(df2_new.blocks.unique())[493:499] #0-3 #246-249 #493-496
    intercept_list = []
    slope_list = []
    mean_intercept_list = []
    mean_slope_list = []
    for i in selection3:
        df3= df2_new[df2_new['blocks']==i].sort_values(by=['CuT_dh_log_noise'])
        #df3= df2[300:600]#.sort_values(by=['CuT_dh'])[0:300]
        X = np.array(df3['CuT_dh_log_noise'])
        Y = np.array(df3['Fe_dh_log_noise'])
        sigma2 = 1
        phi_x = np.vstack([X**0, X**1])
        p=0.2
        V=100.0
        Nsamp = 2000
        logpost_outl = OutlierRegressionLatent(Y, phi_x, sigma2, V, p)
        sampler = MHSampler(lambda theta: np.inf, OutlierGibbsProposal(logpost_outl))
        chain_array = [ ]
        for i in range(4):
            theta0 = np.random.uniform(size=np.sum(phi_x.shape))
            profile_timer(sampler.run, np.array(theta0), Nsamp)
            print("chain.mean, chain.std =", sampler.chain().mean(), sampler.chain().std())
            print("acceptance fraction =", sampler.accept_frac())
            chain_array.append(sampler.chain())
        chain_array = np.array(chain_array)
        flatchain = chain_array.reshape(-1, chain_array.shape[-1])
        #traceplots(chain_array[1], #  xnames=['w0', 'w1', 'w2'],
                    #title="Outlier Regression Weight Traces")
        rho_k, tau = autocorr(chain_array[1], 1000, plot=False)
        print("chain_array.shape =", chain_array.shape)
        print("chain.mean =", flatchain.mean(axis=0))
        print("chain.std =", flatchain.std(axis=0))
        print("tau.shape =", tau.shape)
        Rhat = gelman_rubin(chain_array)
        print("psrf =", Rhat)
        wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))
        flatchain = chain_array.reshape(-1, chain_array.shape[-1])
        #flatchain = flatchain[flatchain[:,-1]==1]
        # flatchain = flatchain[flatchain[:,2]==0]
        func_samples = np.dot(flatchain[:,:2], phi_x)
        mean_intercept = flatchain[:,:1].reshape(-1).mean()
        mean_slope = flatchain[:,1:2].reshape(-1).mean()
        mean_intercept_list.append(mean_intercept)
        mean_slope_list.append(mean_slope)
        intercept_list.append(flatchain[:,:1].reshape(-1))
        slope_list.append(flatchain[:,1:2].reshape(-1))
        
    selection3 = [6,7,8] 
    selection4 = [6,6,6,7,7,7,8,8,8]
    fig,axis = plt.subplots(3,3,figsize=(20,12),sharey=True,sharex=False);
    axis = axis.ravel()
    for i,c in enumerate(selection4):
        c_data = df2_new.loc[df2_new.blocks==c]
        c_data = c_data.reset_index(drop=True)
        if (i-1)%3 ==1:#258
            xvals = np.linspace(c_data['CuT_dh_log_noise'].min(),c_data['CuT_dh_log_noise'].max())
            axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=200,label = 'bore core')
            axis[i].set_title('Block No.' + str(c+1),fontsize=14)
            num=1
            for a_val, b_val in zip(hirearchical_trace_normal_HalfCauchy['alpha'][:,c],hirearchical_trace_normal_HalfCauchy['beta'][:,c]):
                if num==1:
                    axis[i].plot(xvals,a_val+b_val*xvals,'c',alpha=.01,label = 'partial pooling')
                    num+=1
                else:
                    axis[i].plot(xvals,a_val+b_val*xvals,'c',alpha=.01)
        elif (i+1)%3 ==2: #147
            xvals = np.linspace(c_data['CuT_dh_log_noise'].min(),c_data['CuT_dh_log_noise'].max())
            axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=200,label = 'bore core')  
            axis[i].set_title('Block No.' + str(c+1),fontsize=14)
            num=1
            for a_val, b_val in zip(hirearchical_trace_normal_HalfCauchy['alpha'][:,c],hirearchical_trace_normal_HalfCauchy['beta'][:,c]):  
                if num==1:
                    axis[i].plot(xvals,a_val+b_val*xvals,'r',alpha=.01,label = 'no pooling')
                    num+=1
                else:
                    axis[i].plot(xvals,a_val+b_val*xvals,'r',alpha=.01)
        else: ###036
            xvals = np.linspace(c_data['CuT_dh_log_noise'].min(),c_data['CuT_dh_log_noise'].max())
            axis[i].scatter(c_data['CuT_dh_log_noise'],c_data['Fe_dh_log_noise'],color='k',marker='.',s=200,label = 'bore core')  
            axis[i].set_title('Block No.' + str(c+1),fontsize=14)
            num=1
            for j in range(len(selection3)):
                for a_val, b_val in zip(intercept_list[j],slope_list[j]):  
                    if num==1:
                        axis[i].plot(xvals,a_val+b_val*xvals,'b',alpha=.01,label = 'latent outlier')
                        num+=1
                    else:
                        axis[i].plot(xvals,a_val+b_val*xvals,'b',alpha=.01)
        axis[i].legend(loc='lower right',fontsize=14)
        axis[i].set_xlabel('log Cu',fontsize=14)
        axis[i].set_ylabel('log Fe',fontsize=14)
        axis[i].tick_params(axis='both', which='major', labelsize=14)



    #     fig,axis = plt.subplots(1,2,figsize=(10,6),sharey=True,sharex=True);
    #     axis = axis.ravel()
    #     for a_val, b_val in zip(intercept_list[0],slope_list[0]):  
    #         num=1
    #         if num==1:
    #             axis[0].plot(xvals,a_val+b_val*xvals,'b',label = 'latent outlier')
    #             num+=1
    #         else:
    #             axis[0].plot(xvals,a_val+b_val*xvals,'b',alpha=.01)
    #     axis[0].scatter(df2_new.loc[df2_new.blocks==6]['CuT_dh'],df2_new.loc[df2_new.blocks==6]['Fe_dh'],color='k',marker='.',s=200,label = 'bore core')
    #     axis[0].set_xlim(df2_new.loc[df2_new.blocks==6]['CuT_dh'].min(),df2_new.loc[df2_new.blocks==6]['CuT_dh'].max())
    #     #axis[0].set_ylim(df2_new.loc[df2_new.blocks==6]['Fe_dh'].min(),df2_new.loc[df2_new.blocks==6]['Fe_dh'].max())   
        
    #     for a_val, b_val in zip(hirearchical_trace_normal_HalfCauchy['alpha'][:,6],hirearchical_trace_normal_HalfCauchy['beta'][:,6]):  
    #         if num==1:
    #             axis[1].plot(xvals,a_val+b_val*xvals,'r',label = 'no pooling')
    #             num+=1
    #         else:
    #             axis[1].plot(xvals,a_val+b_val*xvals,'r',alpha=.01)
    #     axis[1].scatter(df2_new.loc[df2_new.blocks==6]['CuT_dh'],df2_new.loc[df2_new.blocks==6]['Fe_dh'],color='k',marker='.',s=200,label = 'bore core')          
    #     axis[1].set_xlim(df2_new.loc[df2_new.blocks==6]['CuT_dh'].min(),df2_new.loc[df2_new.blocks==6]['CuT_dh'].max())
    #     #axis[0].set_ylim(df2_new.loc[df2_new.blocks==6]['Fe_dh'].min(),df2_new.loc[df2_new.blocks==6]['Fe_dh'].max())   
        