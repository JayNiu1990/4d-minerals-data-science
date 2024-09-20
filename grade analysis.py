# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:18:20 2022

@author: niu004
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import seaborn as sns
#df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\2min_6min_time_elapse_individual_truck_grade_Oct21_Mar22.csv")
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\all_grade_over_2000tonnage.csv")
counts = df.notnull().sum(axis=1)
i=6
plt.hist(df['all grade over 2000tonnage'][i*1000:(i*1000)+1000],bins=100)
plt.xlim(0,3)  


plt.plot(df['all grade over 2000tonnage'][1000:2000])

plt.figure(figsize=(24,6))
for i in range(0,2,1):
    plt.plot(df['all grade over 2000tonnage'][i*1000:(i*1000)+1000])
plt.ylim(0,3)    
plt.xlabel('time step')
plt.ylabel('4s grade')


df1 = df.groupby(np.arange(len(df))//15).mean()
plt.figure(figsize=(24,6))
for i in range(0,2,1):
    plt.plot(df1['all grade over 2000tonnage'][i*1000:(i*1000)+1000])
plt.ylim(0,3)  
plt.xlabel('time step')
plt.ylabel('1min grade') 

df2 = df.groupby(np.arange(len(df))//450).mean()
plt.figure(figsize=(24,6))
for i in range(0,2,1):
    plt.plot(df2['all grade over 2000tonnage'][i*1000:(i*1000)+1000])
plt.ylim(0,3)  
plt.xlabel('time step')
plt.ylabel('30mins grade')




plt.figure(figsize=(24,6))
plt.plot(df['all grade over 2000tonnage'][0:200],label='4s',color='r')
df1 = df.groupby(np.arange(len(df))//15).mean()
plt.plot(df1['all grade over 2000tonnage'][0:200],label='1min',color='g')
df2 = df.groupby(np.arange(len(df))//450).mean()
plt.plot(df2['all grade over 2000tonnage'][0:200],label='30min',color='b',alpha=0.5)
df3 = df.groupby(np.arange(len(df))//900).mean()
plt.plot(df3['all grade over 2000tonnage'][0:200],label='1h',color='orange',alpha=0.5)
plt.ylim(0,3)  
plt.xlabel('time step')
plt.ylabel('copper grade')
plt.legend(loc='upper right')





df['index'] = df.index
# a = df.iloc[[0]].values
# a = list(a)
df_new = []
for i in range(len(df)):
    a = list(df.iloc[[i]].values)
    a1 = [x for x in a[0] if np.isnan(x) == False]
    a2 = pd.DataFrame(random.sample(a1,int(counts[i]*0.05))).T
    df_new.append(a2)
df_new = pd.concat(df_new)
df["average_grade"] = df.mean(axis=1)
df_new["average_grade"] = df_new.mean(axis=1)
# plt.scatter(df_new["average_grade"],df["average_grade"])
# plt.xlabel('mean grade on 5% truck load',fontsize=12)
# plt.ylabel('mean grade on entire truck load',fontsize=12)
# sns.regplot(
#     df_new["average_grade"][0:300],
#     df["average_grade"][0:300],
#     scatter_kws={"color": "b"},
#     line_kws={"color": "k"},
# )
import random
import numpy
from matplotlib import pyplot

pyplot.hist(df['all grade over 2000tonnage'][0:20000],bins=100,alpha=0.9, label='3000points',density=True)
pyplot.hist(df['all grade over 2000tonnage'][0:2000],bins=100,alpha=0.5, label='2000points',density=True)
pyplot.hist(df['all grade over 2000tonnage'][0:1000],bins=100,alpha=0.5, label='1000points',density=True)
pyplot.legend()
pyplot.show()


    
plt.plot(df['index'][0:500],df['all grade over 2000tonnage'][0:500])
plt.xlabel('time step')
plt.ylabel('mean copper grade from MR sensor')


list_5percentage_grade = []
for i in range(len(df)):
    list_5percentage_grade.append(df[i:i+1][df[i:i+1].columns[0:int(counts[i]*0.50)]])
list_5percentage_grade = pd.concat(list_5percentage_grade)
df["average_grade_50percentage"] = list_5percentage_grade.mean(axis=1)
df_list = []
for n in range(0,3000,100):
    df_list.append(df[n:n+100])
for i in range(3):
    fig,axis = plt.subplots(3,2,figsize=(10,10))
    axis = axis.ravel()
    for j in range(6):
        axis[j].scatter(df_list[6*i:6*i+6][j]['average_grade_50percentage'],df_list[6*i:6*i+6][j]['average_grade'])
        
    fig.text(0.3, 0.08, 'Log CuT', ha='center', va='center') 
    fig.text(0.7, 0.08, 'log CuT', ha='center', va='center')     
    fig.text(0.04, 0.5, 'Log Fe', va='center', rotation='vertical')






plt.figure(figsize=(8,6))
bins = np.arange(0,5,0.1)
plt.hist(df["average_grade"],bins=bins,alpha=0.5)
plt.hist(df["average_grade_5percentage"],bins=bins,alpha=0.5)
plt.xlabel("Data", size=14)
plt.ylabel("Count", size=14)

plt.plot(df.index.values[0:100],df["average_grade"][0:100],alpha=0.5)
plt.plot(df.index.values[0:100],df["average_grade_5percentage"][0:100],alpha=0.5)

list_50percentage_grade = []
for i in range(len(df)):
    list_50percentage_grade.append(df[i:i+1][df[i:i+1].columns[0:int(counts[i]*0.5)]])
list_50percentage_grade = pd.concat(list_50percentage_grade)
df["average_grade_50percentage"] = list_50percentage_grade.mean(axis=1)

plt.plot(df.index.values[0:100],df["average_grade"][0:100],alpha=0.5)
plt.plot(df.index.values[0:100],df["average_grade_50percentage"][0:100],alpha=0.5)


plt.figure(figsize=(8,6))
bins = np.arange(0,5,0.1)
plt.hist(df["average_grade"],bins=bins,alpha=0.5)
plt.hist(df["average_grade_50percentage"],bins=bins,alpha=0.5)
plt.xlabel("Data", size=14)
plt.ylabel("Count", size=14)
df["average_grade_log"] = np.log(df["average_grade"])
# import pymc3 as pm
# with pm.Model() as m1:
#     grade = pm.LogNormal()
import scipy
import matplotlib.pyplot as plt
loc,scale = scipy.stats.distributions.norm.fit(df["average_grade_log"])
x =np.arange(-3,3,0.1)
pdf = scipy.stats.norm.pdf(x, loc, scale)
plt.plot(x,pdf,label = "lognormal fit")
plt.hist(df["average_grade_log"],density=True, bins=100, label="distribution of log mean grade \n for each individual truck")
plt.xlabel("mean grade of individual truck")
plt.ylabel("density")
plt.legend()

plt.plot(df["average_grade_5percentage"][0:100],label="measured 5% truck load")
plt.plot(df["average_grade_50percentage"][0:100],label="measured 50% truck load")
plt.plot(df["average_grade"][0:100],label="measured 100% truck load")
plt.xlabel("time step")
plt.ylabel("mean grade")
plt.legend()

#plt.hist(df["average_grade"],density=True, bins=100, label="distribution of mean grade \n for each individual truck")
plt.xlabel("mean grade of individual truck")
plt.ylabel("density")
plt.legend()


# plt.plot(x, scipy.stats.norm.pdf(x, post_pred["mu"].mean(0), post_pred["sigma"].mean(0)))
# plt.hist(df["average_grade_log"],bins=100,alpha=0.5,density=True)
# ax = plt.subplots(1, 1, sharey=True)
# for i in range(200):
#     x =np.arange(-1,1,0.1)
#     plt.plot(x,scipy.stats.norm.pdf(x, post_pred["mu"][i], post_pred["sigma"][i]))
plt.hist(df[500:1000],bins=50)
plt.xlim(0,3)

import pymc3 as pm
import arviz as az
data = np.random.randn(100)
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    sd = pm.HalfNormal("sd", sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=sd, observed=data)

    idata = pm.sample(return_inferencedata=True)

with model:
    post_pred = pm.sample_posterior_predictive(idata.posterior)
# add posterior predictive to the InferenceData
az.concat(idata, az.from_pymc3(posterior_predictive=post_pred), inplace=True)
fig, ax = plt.subplots()
az.plot_ppc(idata, ax=ax)
#ax.axvline(data.mean(), ls="--", color="r", label="True mean")
ax.legend(fontsize=10);







######################lognormal##############################
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Kansanshi Bore Core\kmp_ddh_mra.csv")
df = df.dropna()
df = df[df['CU']>0.1]
param_bore_core = scipy.stats.distributions.lognorm.fit(df["CU"])
x=np.linspace(0,5,100)
pdf_fitted = scipy.stats.lognorm.pdf(x, param_bore_core[0], loc=param_bore_core[1], scale=param_bore_core[2])
plt.plot(x,pdf_fitted,'r-')
plt.hist(df["CU"],bins=x,density=True)
plt.xlabel("grade from bore core")
plt.ylabel("density")

df1 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\2min_6min_time_elapse_individual_truck_grade_Oct21_Mar22.csv")
df1["average_grade"] = df1.mean(axis=1)
param_truck = scipy.stats.distributions.lognorm.fit(df1["average_grade"])
x=np.linspace(0,5,100)
pdf_fitted1 = scipy.stats.lognorm.pdf(x, param_truck[0], loc=param_truck[1], scale=param_truck[2])
plt.plot(x,pdf_fitted1,'r-')
plt.hist(df1["average_grade"],bins=x,density=True)
plt.xlabel("grade from truck")
plt.ylabel("density")


df_belt = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\all_grade_over_2000tonnage.csv")
counts = df_belt.notnull().sum(axis=1)
import pymc3 as pm
import arviz as az
with pm.Model() as model_bore_core:
    mu = pm.Normal("mu",mu=param_bore_core[1],sigma = 100)
    sigma = pm.Normal("sigma",mu=param_bore_core[2], sigma = 100)
    obs = pm.LogNormal("obs",mu = mu, sigma = sigma, observed = df_belt['all grade over 2000tonnage'][0:1000])
    trace_bore_core = pm.sample(return_inferencedata=True)
#az.plot_trace(trace,compact=False,figsize = (20, 8))
# az.summary(trace)
with model_bore_core:
    post_pred_bore_core = pm.sample_posterior_predictive(trace_bore_core.posterior)
az.concat(trace_bore_core, az.from_pymc3(posterior_predictive=post_pred_bore_core), inplace=True)


import pymc3 as pm
import arviz as az
with pm.Model() as model_truck:
    mu = pm.Normal("mu",mu=1,sigma = 100)
    sigma = pm.Normal("sigma",mu=0, sigma = 100)
    obs = pm.Normal("obs",mu = mu, sigma = sigma, observed = df_belt['all grade over 2000tonnage'][0:1000])
    trace_truck = pm.sample(return_inferencedata=True)
#az.plot_trace(trace,compact=False,figsize = (20, 8))
# az.summary(trace)
with model_truck:
    post_pred_truck = pm.sample_posterior_predictive(trace_truck.posterior)
az.concat(trace_truck, az.from_pymc3(posterior_predictive=post_pred_truck), inplace=True)

fig, ax = plt.subplots()
az.plot_ppc(trace_bore_core, ax=ax)
#ax.plot(x,pdf_fitted,'r-')
#plt.hist(df_modelling,bins=100,density=True)
#ax.axvline(df_modelling.mean(), ls="--", color="r", label="True mean")
ax.legend(fontsize=10);
plt.xlim(0,3)























import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
df_mini = pd.read_csv(r"C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\PLC_CMP_Data_For02D03M2022Y.csv")
df_mini = df_mini.loc[(df_mini["GRADE"]>0) & (df_mini["TPH"]>2000)]
df_mini['GRADE_1'] = df_mini['GRADE']/100
df_modelling = df_mini["GRADE_1"][0:2000].to_frame()
df_modelling['TIMECOUNT'] = df_mini['TIMECOUNT']
with pm.Model() as model:
    mu = pm.Normal("mu",mu=0.8,sigma = 100)
    sigma = pm.Normal("sigma",mu=1, sigma = 100)
    obs = pm.LogNormal("obs",mu = mu, sigma = sigma, observed = df_modelling['GRADE_1'])
    
    trace = pm.sample(return_inferencedata=True)
import arviz as az
with model:
    post_pred = pm.sample_posterior_predictive(trace.posterior)

az.concat(trace, az.from_pymc3(posterior_predictive=post_pred), inplace=True)
fig, ax = plt.subplots()
az.plot_ppc(trace, ax=ax)
#plt.hist(df_modelling,bins=100,density=True)
#ax.axvline(df_modelling.mean(), ls="--", color="r", label="True mean")
ax.legend(fontsize=10);
plt.xlim(0,3)












import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
df_mini = pd.read_csv(r"C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\PLC_CMP_Data_For02D03M2022Y.csv")
df_mini = df_mini.loc[(df_mini["GRADE"]>0) & (df_mini["TPH"]>2000)]
df_mini['GRADE_1'] = df_mini['GRADE']/100

df_modelling = np.array(df_mini["GRADE_1"][0:100].to_frame())[:,0]
with pm.Model() as model:
    weights = pm.Dirichlet("w", np.ones(3))
    #mu = pm.Normal("mu",mu=0.8,sigma = 100,shape=3)
    #sigma = pm.Normal("sigma",mu=1, sigma = 100,shape=3)
    mu = pm.Normal(
    "mu",
    np.zeros_like(weights),
    1.0,
    shape=3,
    transform=pm.transforms.ordered,
    testval=[1, 2, 3],
    )
    tau = pm.Gamma("tau", 1.0, 1.0, shape=3)

    obs = pm.NormalMixture("obs",w= weights,mu = mu, tau = tau, observed = df_modelling)

import arviz as az
with model:
    trace = pm.sample(return_inferencedata=True)
    post_pred = pm.sample_posterior_predictive(trace)
trace.add_groups(posterior_predictive=post_pred)

#az.concat(trace, az.from_pymc3(posterior_predictive=post_pred), inplace=True)

az.plot_ppc(trace)
#plt.hist(df_modelling,bins=100,density=True)
#ax.axvline(df_modelling.mean(), ls="--", color="r", label="True mean")
ax.legend(fontsize=10);

import arviz as az
import numpy as np
import pymc3 as pm

from matplotlib import pyplot as plt
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")
N = 1000
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
df_mini = pd.read_csv(r"C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\PLC_CMP_Data_For02D03M2022Y.csv")
df_mini = df_mini.loc[(df_mini["GRADE"]>0) & (df_mini["TPH"]>2000)]
df_mini['GRADE_1'] = df_mini['GRADE']/100
#######################################Gaussian Mixture Model###################################################
df_modelling = np.array(df_mini["GRADE_1"][0:200].to_frame())[:,0]
fig, ax = plt.subplots(figsize=(8, 6))
# ax.hist(x, bins=30, density=True, lw=0);

ax.hist(df_modelling, bins=30, density=True, lw=0);

W = np.array([0.35, 0.4,0.25])

MU = np.array([0.1, 0.3, 0.5])
SIGMA = np.array([0.01, 0.1,0.2])
component = np.random.choice(MU.size, size=N, p=W)
x = np.random.normal(MU[component], SIGMA[component], size=N)
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(x, bins=30, density=True, lw=0);
with pm.Model() as model:
    w = pm.Dirichlet("w", np.ones_like(W))

    mu = pm.Normal(
        "mu",
        np.zeros_like(W),
        1,
        shape=W.size,
        transform=pm.transforms.ordered,
        testval=[0.4, 0.6, 0.8],
    )
    #tau = pm.Gamma("tau", 2, 1.0, shape=W.size)
    sigma = pm.Uniform('tau', lower=0, upper=1,shape=W.size)
    x_obs = pm.NormalMixture("x_obs", w, mu, sigma=sigma, observed=x)
with model:
    trace = pm.sample(400, n_init=10000, tune=200, return_inferencedata=True)

    # sample posterior predictive samples
    ppc_trace = pm.sample_posterior_predictive(trace, var_names=["x_obs"], keep_size=True)

trace.add_groups(posterior_predictive=ppc_trace)
import arviz as az
fig, ax = plt.subplots()
plt.xlim(0, 2)
ax = az.plot_ppc(trace,observed=True, ax=ax)

######################GP###############################################
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import pymc3 as pm
# import theano.tensor as tt
# df_mini = pd.read_csv(r"C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\PLC_CMP_Data_For02D03M2022Y.csv")
# df_mini = df_mini.loc[(df_mini["GRADE"]>0) & (df_mini["TPH"]>2000)]
# df_mini['GRADE_1'] = df_mini['GRADE']/100
# df_modelling = df_mini["GRADE_1"][0:600].to_frame()
# df_modelling['TIMECOUNT'] = df_mini['TIMECOUNT'] 
# df_modelling.reset_index(drop=True,inplace=True)
# df_modelling['index'] = df_modelling.index

# X = np.array(df_modelling['index'])[:,None]
# y = np.array(df_modelling['GRADE_1'])

# # set the seed
# ## Plot the data and the unobserved latent function
# fig = plt.figure(figsize=(12,5)); ax = fig.gca()
# ax.plot(X, y, 'ok', ms=3, alpha=0.5, label="Data");
# ax.set_xlabel("X"); ax.set_ylabel("The true f(x)"); plt.legend();

# Z = np.linspace(0,600,600)[:,None]

# with pm.Model() as model:
#     # priors on the covariance function hyperparameters
#     l = pm.Uniform('l', 0, 10)

#     # uninformative prior on the function variance
#     log_s2_f = pm.Uniform('log_s2_f', lower=-10, upper=5)
#     s2_f = pm.Deterministic('s2_f', tt.exp(log_s2_f))

#     # uninformative prior on the noise variance
#     log_s2_n = pm.Uniform('log_s2_n', lower=-10, upper=5)
#     s2_n = pm.Deterministic('s2_n', tt.exp(log_s2_n))

#     # covariance functions for the function f and the noise
#     f_cov = s2_f * pm.gp.cov.ExpQuad(1, l)
#     gp = pm.gp.Marginal(cov_func=f_cov)
    
#     y_obs = gp.marginal_likelihood('y_obs',X=X,y=y, noise=s2_n)
#     mp = pm.find_MAP()
# with model:
#     f_pred = gp.conditional("f_pred", Z)
# with model:
#     pred_samples = pm.sample_posterior_predictive([mp], var_names=['f_pred'], samples=2000)


# # plot the results
# fig = plt.figure(figsize=(12,5)); ax = fig.gca()
# # plot the samples from the gp posterior with samples and shading
# from pymc3.gp.util import plot_gp_dist
# plot_gp_dist(ax, pred_samples["f_pred"], Z);

# # plot the data and the true latent function
# plt.plot(X, y, 'ok', ms=3, alpha=0.5, label="Observed data");

# # axis labels and title
# plt.xlabel("X"); plt.ylim([0,2]);
# plt.title("Posterior distribution over $f(x)$ at the observed values"); plt.legend();
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
#energy_df = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\energydata_complete.csv')
df1 = pd.read_csv(r'C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\grade_over_2000tonnage_each_row_grade\PLC_CMP_Data_For01D11M2021Y.csv')
df2 = pd.read_csv(r'C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\grade_over_2000tonnage_each_row_tonnage\PLC_CMP_Data_For01D11M2021Y.csv')
df = pd.concat([df1,df2],axis=1)

df['tonnage'] = 4*(df2['tonnage over 2000tonnage '])/3600
df = df.drop('tonnage over 2000tonnage ',axis=1)
df = df.loc[df['grade over 2000tonnage ']>0]


df.loc[df['grade over 2000tonnage ']<=0.8,'sorting'] = 0
df.loc[df['grade over 2000tonnage ']>0.8,'sorting'] = 1
df = df.reset_index(drop=True)

df = df[df['sorting']==1]
print(df['grade over 2000tonnage '].mean())
print(df['tonnage'].sum())

plt.plot(df['tonnage'] ,color='blue')
plt.xlabel('Time step')
plt.ylabel('Mean copper grade from MR belt sensor (4s)')
plt.xlim(0,500)

plt.plot(df['grade over 2000tonnage '] ,color='blue')
plt.xlabel('Time step')
plt.ylabel('Mean copper grade from MR belt sensor (4s)')
plt.xlim(0,500)




df1 = pd.read_csv(r'C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\grade_over_2000tonnage_each_row_grade\PLC_CMP_Data_For09D11M2021Y.csv')
df1 = df1.groupby(np.arange(len(df1))//1).mean()
plt.subplots(1,1,figsize=(12,4))
plt.plot(df1['grade over 2000tonnage '],color='blue')
plt.xlabel('Time step')
plt.ylabel('Mean copper grade from MR belt sensor (4s)')




import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
arr = os.listdir('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\grade_over_2000tonnage_each_row_grade\\')
mean_grade_per_day=[]
for i in arr:
    df = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\grade_over_2000tonnage_each_row_grade\\'+str(i))
    if len(df)>10:
        df = df.loc[df['grade over 2000tonnage ']>0]
        #df = df.reset_index(drop=True)
        mean_grade_per_day.append(df['grade over 2000tonnage '].mean())


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
cube = np.ones((3,6,3),dtype='bool')
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.voxels(cube,facecolor='#E02050',edgecolors='k')
ax.axis('off')
plt.show() 


import matplotlib.pyplot as plt
import numpy as np

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


# prepare some coordinates
x, y, z = np.indices((6, 6, 6))

# draw cuboids in the top left and bottom right corners, and a link between them
cube1 = (x < 3) & (y < 3) & (z < 1)
cube2 = (x < 3) & (y < 3) & ( z > 2)& ( z < 2)
#cube3 = (x < 2) & (y < 3) & (z >= 3)

# combine the objects into a single boolean array
voxels = cube1 | cube2

# set the colors of each object
colors = np.empty(voxels.shape, dtype=object)
colors[cube1] = 'blue'
colors[cube2] = 'green'

# and plot everything
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(voxels, facecolors=colors, edgecolor='k')

plt.show()


import skimage.io as io
img = io.imread('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\edsr_12.png')
io.imsave('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\edsr_121.png',img[21:51,85:115,:]) #img[256:616,1024:1384,:])








