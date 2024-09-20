import matplotlib.pyplot as plt
import matplotlib.cm as cmap
# cm = cmap.inferno

import numpy as np
print(np.__version__)
import scipy as sp
import theano
# theano.config.optimizer = 'None'
import theano.tensor as tt
import theano.tensor.nlinalg
import sys
# sys.path.insert(0, "../../..")
import pymc3 as pm
# np.random.seed(20090425)
import pandas as pd
import theano.tensor as tt
import random

data = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\multiscale.csv")

X = np.array(data['x']).reshape(-1,1)
y = np.array(data['y'])

# fig = plt.figure(figsize=(14,5)); ax = fig.add_subplot(111)
# ax.plot(X, y, 'ok', ms=10);
# ax.set_xlabel("partial");
# ax.set_ylabel("entire");


# np.random.seed(20090425)
# n = 20
# X = np.sort(3*np.random.rand(n))[:,None]

# with pm.Model() as model:
#     # f(x)
#     l_true = 0.3
#     s2_f_true = 1.0
#     cov = s2_f_true * pm.gp.cov.ExpQuad(1, l_true)

#     # noise, epsilon
#     s2_n_true = 0.1
#     K_noise = s2_n_true**2 * tt.eye(n)
#     K = cov(X) + K_noise

# # evaluate the covariance with the given hyperparameters
# K = theano.function([], cov(X) + K_noise)()

# # generate fake data from GP with white noise (with variance sigma2)
# y = np.random.multivariate_normal(np.zeros(n), K)

# with pm.Model() as model:
#     ℓ = pm.Uniform("ℓ", lower=0,upper=2.5)
#     η =  pm.Uniform("η", lower=0,upper=0.5)
#     cov = η ** 2 * pm.gp.cov.Matern52(1, ℓ)
#     gp = pm.gp.Marginal(cov_func=cov)
#     σ = pm.HalfCauchy("σ", beta=5)
#     y_ = gp.marginal_likelihood("y", X=df_train_x, y=df_train_y, noise=σ)
#     mp = pm.find_MAP()

with pm.Model() as model:
    # priors on the covariance function hyperparameters
    l = pm.Uniform('l',0, 10)
    # uninformative prior on the function variance
    log_s2_f = pm.Uniform('log_s2_f', lower=-10, upper=10)
    s2_f = pm.Deterministic('s2_f', tt.exp(log_s2_f))

    # uninformative prior on the noise variance
    log_s2_n = pm.Uniform('log_s2_n', lower=-10, upper=10)
    s2_n = pm.Deterministic('s2_n', tt.exp(log_s2_n))
    
    #cov = s2_f * pm.gp.cov.ExpQuad(1, l)
    cov =  s2_f*pm.gp.cov.Exponential(1, l)
    #cov = s2_f * pm.gp.cov.Matern32(1,l)
    
    gp = pm.gp.Marginal(cov_func=cov)

    y_ = gp.marginal_likelihood("y", X=X, y=y, noise=s2_n)

    mp = pm.find_MAP()

with model:
    X_new = np.arange(-1,7,0.005)[:,None]
    f_pred = gp.conditional("f_pred", X_new)
    pred_samples = pm.sample_posterior_predictive([mp], var_names=['f_pred'], samples=2000)
    
mra = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\multiscale_mra1.csv")
x1 = np.array(mra['x'])
y1 = np.array(mra['y'])

# fig = plt.figure(figsize=(12, 5))
# plt.plot(X,y, "ok", ms=8,  label="Bore core associated data")
# plt.scatter(x1,y1,color='blue', label="mra data")
# plt.xlabel("log10 tonnage")
# plt.ylabel("log10 variance")
# plt.title("Gaussian process")
# plt.legend()

# plot the results
fig = plt.figure(figsize=(12, 5))
ax = fig.gca()
# plot the samples from the gp posterior with samples and shading
from pymc3.gp.util import plot_gp_dist
plot_gp_dist(ax, pred_samples["f_pred"], X_new)

mean_pred = pred_samples["f_pred"].mean(axis=0)

# Plot the mean of the posterior predictive distribution
plt.plot(X_new, mean_pred, color='black', lw=4, label='Mean Posterior GP')
# plot the data and the true latent function
plt.plot(X,y, "ok", ms=8,  label="Bore core associated data")
plt.scatter(x1,y1,color='blue', label="mra data")
# axis labels and title
plt.xlabel("log10 tonnage scale")
plt.ylabel("log10 variance")
plt.title("Gaussian process")
plt.legend()





variable = pred_samples['f_pred'][:,283]

# Calculate the mean and standard deviation
mean_pred_x_new = np.mean(variable)
std_pred_x_new = np.std(variable)

import scipy.stats as stats
# Calculate the cumulative probability up to x_0
x_0 = -0.521
cdf_value = stats.norm.cdf(x_0, loc=mean_pred_x_new, scale=std_pred_x_new)

# Calculate the probability of being higher than x_0
prob_higher_than_x0 = 1 - cdf_value
# Create a range of values for plotting the Gaussian distribution
x_values = np.linspace(min(variable), max(variable), 1000)
gaussian_pdf = stats.norm.pdf(x_values, loc=mean_pred_x_new, scale=std_pred_x_new)
# Plotting the marginal distribution
plt.figure(figsize=(16, 12))
plt.hist(variable, bins=30, density=True, alpha=0.6, color='b')
plt.axvline(mean_pred_x_new, color='r', linestyle='--', label=f"Mean: {mean_pred_x_new:.3f}")
plt.axvline(mean_pred_x_new + std_pred_x_new, color='g', linestyle='--', label=f"Mean + 1 Std")
plt.axvline(mean_pred_x_new - std_pred_x_new, color='g', linestyle='--', label=f"Mean - 1 Std")
plt.plot(x_values, gaussian_pdf, 'r-', label=f'Gaussian fit')
plt.axvline(-0.521, color='black', linestyle='--', label="Ground truth")
plt.tick_params(axis='both', which='major', labelsize=18) 
plt.xlabel("Predicted Variance at MRA belt sensor scale",fontsize=24)
plt.ylabel("Density",fontsize=24)
# plt.title(f"Marginal Distribution at x=0.415",fontsize=24)
plt.legend(fontsize=20)
plt.show()

from sklearn.metrics import mean_squared_error 
mean_squared_error([-0.386],[-0.521]) 
















