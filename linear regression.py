import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\kmp_ddh_mra.csv')
df = df.dropna()
df = df[(pd.to_numeric(df["X"], errors='coerce')>=3000)& (pd.to_numeric(df["X"], errors='coerce')<4000)
        & (pd.to_numeric(df["Y"], errors='coerce')>=12500)& (pd.to_numeric(df["Y"], errors='coerce')<13000)
        & (pd.to_numeric(df["Z"], errors='coerce')>=1200)& (pd.to_numeric(df["Z"], errors='coerce')<1400)]
df = df[df['STRAT']!='DUMP']
# strat = df.groupby('STRAT').size()
# strat = strat.to_frame()
# plt.hist(df['STRAT'],bins=10) 
df['X'] = round(df['X'],2)
df['Y'] = round(df['Y'],2)
df['Z'] = round(df['Z'],2)
n = 100
m = 100
xx1 = np.arange(df["X"].min(), df["X"].max(), n).astype('float64')
yy1 = np.arange(df["Y"].min(), df["Y"].max(), n).astype('float64')
zz1 = np.arange(df["Z"].min(), df["Z"].max(), m).astype('float64')

blocks = []
for k in zz1:
    for j in yy1:
        for i in xx1:
            sub_block = df.loc[(pd.to_numeric(df["X"], errors='coerce')>=i) & (pd.to_numeric(df["X"], errors='coerce')<i+n) &
                         (pd.to_numeric(df["Y"], errors='coerce')>=j) & (pd.to_numeric(df["Y"], errors='coerce')<j+n)
                         &(pd.to_numeric(df["Z"], errors='coerce')>=k) & (pd.to_numeric(df["Z"], errors='coerce')<k+m)]
            blocks.append(sub_block)
blocks1 = []
for i,j in enumerate(blocks):
    if len(j)>=5:
        blocks1.append(j)
for i, j in enumerate(blocks1):
    blocks1[i]['block'] = i
 
df2_new = pd.concat(blocks1)   
n_blocks = len(df2_new['block'].unique())
import pymc3 as pm
import theano

block_idx = df2_new.block.values
block_names = df2_new.block.unique()
n_blocks = len(df2_new.block.unique())
df2_new['STRAT_CODE'] = df2_new.STRAT.factorize()[0]

with pm.Model() as hierarchical_model:
    alpha = pm.Normal('alpha',0,sd=10,shape=n_blocks)
    beta1 = pm.Normal('beta1',0,sd=10,shape=n_blocks)
    eps = pm.HalfCauchy('eps',10)
    Cu_est = alpha[block_idx] +beta1[block_idx]*df2_new.VEIN.values
    y = pm.Normal('y',mu = Cu_est,sd=eps,observed = df2_new.CU)
    
    
with hierarchical_model:
    hierarchical_trace = pm.sample(2000)

pm.traceplot(hierarchical_trace)    

selection = [0,1,2]
fig, axis = plt.subplots(1,3,figsize=(12,6),sharey=True, sharex = True);
axis = axis.ravel();
for i, c in enumerate(selection):
    c_data = df2_new[df2_new['block']==i]
    
    # c_data = df2_new[df2_new.block==c]
    # c_data = c_data.reset_index(drop=True)
    # block_name = df2_new.block.unique()
    # c_index = np.where(block_name==c)[0][0]
    
    xvals = np.linspace(-0.2,5.2)
    for a,b1,b2 in zip(hierarchical_trace['alpha'][1000:,i],hierarchical_trace['beta1'][1000:,i]):
        axis[i].plot(xvals,a+b1*xvals+b2*xvals,'b',alpha=0.1)
    axis[i].scatter(c_data.VEIN + np.random.randn(len(c_data))*0.01,c_data.CU,alpha=1,color='k',marker='.',s=80,label='bore core data')
    
    
    
    
    
    
    