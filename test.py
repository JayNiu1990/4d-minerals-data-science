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
    
    # cov = s2_f * pm.gp.cov.ExpQuad(1, l)
    cov =  s2_f*pm.gp.cov.Exponential(1, l)
    #cov = s2_f * pm.gp.cov.Matern32(1,l)
    
    gp = pm.gp.Marginal(cov_func=cov)

    y_ = gp.marginal_likelihood("y", X=X, y=y, noise=s2_n)

    mp = pm.find_MAP()

with model:
    X_new = np.arange(-1,7,0.5)[:,None]
    f_pred = gp.conditional("f_pred", X_new)
    pred_samples = pm.sample_posterior_predictive([mp], var_names=['f_pred'], samples=2000)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(X, y, 'ok', ms=10, label="Observed data")
plt.plot(X_new, np.mean(pred_samples['f_pred'], axis=0), label="Mean prediction")
for i in range(pred_samples['f_pred'].shape[0]):
    plt.plot(X_new, pred_samples['f_pred'][i], ls='--', label=f"Sample {i+1}")
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("GP Regression with Exponential Kernel")
plt.show()