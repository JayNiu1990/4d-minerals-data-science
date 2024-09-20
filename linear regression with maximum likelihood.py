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
from scipy import stats
fields = ['BHID','Fe_dh','As_dh','CuT_dh',"X","Y","Z","LITH","AL_ALT"]
pio.renderers.default='browser'
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)
df = df.dropna()
df = df[(pd.to_numeric(df["CuT_dh"], errors='coerce')>0) & (pd.to_numeric(df["Fe_dh"], errors='coerce')>0)& (pd.to_numeric(df["As_dh"], errors='coerce')>0)
        & (pd.to_numeric(df["X"], errors='coerce')>=17000)& (pd.to_numeric(df["X"], errors='coerce')<17500)
        & (pd.to_numeric(df["Y"], errors='coerce')>=107000)& (pd.to_numeric(df["Y"], errors='coerce')<107500)
        & (pd.to_numeric(df["Z"], errors='coerce')>=2500)& (pd.to_numeric(df["Z"], errors='coerce')<3000)]

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

df['CuT_dh_log'] = stats.zscore(df['CuT_dh'])#np.log(df['CuT_dh'])
df['CuT_dh_log'] = round(df['CuT_dh_log'],3)

df['Fe_dh_log'] = stats.zscore(df['Fe_dh'])#np.log(df['Fe_dh'])
df['Fe_dh_log'] = round(df['Fe_dh_log'],3)

df['As_dh_log'] = stats.zscore(df['As_dh'])#np.log(df['As_dh'])
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

df2['CuT_dh_zscore'] = stats.zscore(df2['CuT_dh'])
df2['Fe_dh_zscore'] = stats.zscore(df2['Fe_dh'])

#df2.groupby(['LITH']).size()
# fig = px.scatter_3d(df2, x="X",y="Y",z="Z",color="Fe")
# fig.update_traces(marker_size=2)
# fig.update_layout(font=dict(size=14))
# fig.update_layout(scene_aspectmode='data')
# fig.show()    

#df2 = df2.loc[df2['LITH']==31]
#df2 = df2.reset_index(drop=True)
n = 50
xx1 = np.arange(17000, 17500, n).astype('float64')
yy1 = np.arange(107000, 107500, n).astype('float64')
zz1 = np.arange(2500, 3000, n).astype('float64')

blocks = []
for k in zz1:
    for j in yy1:
        for i in xx1:
            sub_block = df2.loc[(pd.to_numeric(df2["X"], errors='coerce')>=i) & (pd.to_numeric(df2["X"], errors='coerce')<i+n) &
                         (pd.to_numeric(df2["Y"], errors='coerce')>=j) & (pd.to_numeric(df2["Y"], errors='coerce')<j+n)
                         &(pd.to_numeric(df2["Z"], errors='coerce')>=k) & (pd.to_numeric(df2["Z"], errors='coerce')<k+n)]
            blocks.append(sub_block)
blocks1 = []
for i,j in enumerate(blocks):
    if len(j)>=5:
        blocks1.append(j)
for i, j in enumerate(blocks1):
    blocks1[i]['blocks'] = i
    
df2_new = pd.concat(blocks1)    


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
        plt.setp(ax_hist.get_xticklabels(), visible=False)
        plt.setp(ax_hist.get_yticklabels(), visible=False)
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
        logL = -0.5*np.sum(resids**2/S2 + log(2*np.pi*S2))
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
        ##### w (2x1)  q (26,)

        # Step 1:  propose from P(q|w)
        # Each mixture component has a Gaussian variance which may be inflated
        # The final log likelihood sums over the log likelihoods for each point
        # Conditioned on the w's, figure out the density ratios for q
        resids = self.y - np.dot(w, self.phi_x)    ##resids (26)

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
# Now try to sample and see what breaks
# The way we've coded it, the Proposal has all the model information,
# so we can just use a dummy Model and let the Sampler automatically accept

df3= blocks1[11].sort_values(by=['CuT_dh_zscore'])
#df3= df2[300:600]#.sort_values(by=['CuT_dh'])[0:300]
X = np.array(df3['CuT_dh_zscore'])
Y = np.array(df3['Fe_dh_zscore'])
sigma2 = 1
phi_x = np.vstack([X**0, X**1])
p=0.2
V=100.0
Nsamp = 2000
logpost_outl = OutlierRegressionLatent(Y, phi_x, sigma2, V, p)
sampler = MHSampler(lambda theta: np.inf, OutlierGibbsProposal(logpost_outl))

chain_array = []
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
# print("chain_array.shape =", chain_array.shape)
# print("chain.mean =", flatchain.mean(axis=0))
# print("chain.std =", flatchain.std(axis=0))
# print("tau.shape =", tau.shape)
# Rhat = gelman_rubin(chain_array)
# print("psrf =", Rhat)
wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))

plt.figure(figsize=(6,4))
plt.plot(X, Y, ls='None', marker='o', ms=3, label="Data")
plt.plot(X, np.dot(wML, phi_x), ls='--', lw=2, label="Maximum Likelihood")

flatchain = chain_array.reshape(-1, chain_array.shape[-1])
func_samples = np.dot(flatchain[:,:2], phi_x)

post_mu = np.mean(func_samples, axis=0)
post_sig = np.std(func_samples, axis=0)
plt.plot(X, post_mu, ls='--', lw=2, color='dodgerblue', label="Posterior Mean")
plt.fill_between(X, post_mu-post_sig, post_mu+post_sig, color='dodgerblue', alpha=0.5, label="Posterior Variance")
plt.legend(loc='best')
plt.xlabel("Cu grade - Zscore")
plt.ylabel("Fe grade - Zscore")
plt.title("Regression with Outliers (Latent) for Block No.12 bore core data")

# xvals = np.linspace(X.min() - (X.max()-X.min())*0.5,X.max() + (X.max()-X.min())*0.5)
# intercept,slope = flatchain[:,:1].reshape(-1),flatchain[:,1:2].reshape(-1)

# plt.plot(X, Y, ls='None', color = 'black', marker='o', ms=5, label="Data")
# plt.plot(X, post_mu, ls='--', lw=2, color='dodgerblue', label="Posterior Mean")
# # for i, j in zip(intercept,slope):
#     plt.plot(xvals,i+j*xvals,'c',label = 'latent',alpha=.01)



###########gaussian mixture model#########    
sigma2 = 1
phi_x = np.vstack([X**0, X**1])
p=0.2
V=100.0
Nsamp = 20000
logpost_outl = OutlierRegressionMixture(Y, phi_x, sigma2, V, p)
sampler = MHSampler(logpost_outl, GaussianProposal([0.5,0.5]))
chain_array = [ ]
for i in range(5):
    theta0 = np.random.uniform(size=2)
    profile_timer(sampler.run, np.array(theta0), Nsamp)
    print("chain.mean, chain.std =", sampler.chain().mean(), sampler.chain().std())
    print("acceptance fraction =", sampler.accept_frac())
    chain_array.append(sampler.chain())

chain_array = np.array(chain_array)
traceplots(chain_array[1], xnames=['w0', 'w1','w2'],
            title="Outlier Regression Weight Traces")
rho_k, tau = autocorr(chain_array[1], 1000, plot=False)
print("chain_array.shape =", chain_array.shape)
print("tau.shape =", tau.shape)
Rhat = gelman_rubin(chain_array)
print("psrf =", Rhat)
wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))
# Visualize our answer!
plt.figure(figsize=(6,4))
plt.plot(X, Y, ls='None', marker='o', ms=3, label="Data")
plt.plot(X, np.dot(wML, phi_x), ls='--', lw=2, label="Maximum Likelihood")
flatchain = chain_array.reshape(-1, chain_array.shape[-1])
func_samples = np.dot(flatchain, phi_x)
post_mu = np.mean(func_samples, axis=0)
post_sig = np.std(func_samples, axis=0)
plt.plot(X, post_mu, ls='--', lw=2, color='dodgerblue', label="Posterior Mean")
plt.fill_between(X, post_mu-post_sig, post_mu+post_sig, color='dodgerblue', alpha=0.5, label="Posterior Variance")
plt.legend(loc='best')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Regression with Outliers (Mixture)")

