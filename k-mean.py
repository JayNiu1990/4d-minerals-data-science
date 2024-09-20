import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\kmp_ddh_mra.csv", skipinitialspace=True)#, usecols=fields)
df = df.dropna()

df = df[(pd.to_numeric(df["X"], errors='coerce')>=3000)& (pd.to_numeric(df["X"], errors='coerce')<3800)
        & (pd.to_numeric(df["Y"], errors='coerce')>=12500)& (pd.to_numeric(df["Y"], errors='coerce')<13000)&
        (pd.to_numeric(df["Z"], errors='coerce')>=1200)& (pd.to_numeric(df["Z"], errors='coerce')<1400)]
df = df.reset_index(drop=True)
df = df.sample(frac=1)
df = df.reset_index(drop=True)

df1 = df[['X','Y','Z','CU']]
fig = px.scatter_3d(df1, x="X",y="Y",z="Z",color="CU")
fig.update_traces(marker_size=2)
fig.update_layout(font=dict(size=14))
fig.update_layout(scene_aspectmode='data')
fig.show()  
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
normalized_df1 = (df1 - df1.mean()) / df1.std()

from sklearn.metrics import silhouette_score
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_df1)
    silhouette_scores.append(silhouette_score(normalized_df1, cluster_labels))

# Plot the silhouette scores
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.show()








# Specify the number of clusters (you can adjust this)
num_clusters = 8

# Apply k-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

df1['cluster'] = kmeans.fit_predict(normalized_df1)

import plotly.graph_objs as go
import plotly.io as pio

fields = ['X','Y','Z','CU','BHID']
fig = px.scatter_3d(df1, x="X",y="Y",z="Z",color="cluster")
fig.update_traces(marker_size=2)
fig.update_layout(font=dict(size=14))
fig.update_layout(scene_aspectmode='data')
fig.show()  


