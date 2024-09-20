###########skgstat variogram##############
# import warnings
# import matplotlib.pyplot as plt
# import plotly.express as px
# import numpy as np
# import pandas as pd    
# import plotly.io as pio
# import plotly.graph_objs as go
# from scipy import stats
# import skgstat as skg
# from skgstat import Variogram, OrdinaryKriging
# from scipy.optimize import curve_fit
# from skgstat import models
# fields = ['BHID','FROM','TO','LENGTH',"X","Y","Z","CU"]
# pio.renderers.default='browser'
# df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\kmp_ddh_mra.csv", skipinitialspace=True, usecols=fields)
# df = df.dropna(subset=['CU'])
# #number = df['BHID'].unique()
# data = df[df['BHID']=='DCDD001']
# data = data.sort_values('Z', ascending=True)
# # Calculation variogram
# V = skg.Variogram(list(zip(data.Z)), data.CU.values,normalize=False, n_lags=150, maxlag=None, model='spherical')
# #V.plot()
# # plt.scatter(data['Z'],data['CU'])

    
    
import warnings
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd    
import plotly.io as pio
import plotly.graph_objs as go
from scipy import stats
import skgstat as skg
from skgstat import Variogram, OrdinaryKriging
import geostatspy.GSLIB as GSLIB                       # GSLIB utilies, visualization and wrapper
import geostatspy.geostats as geostats                 # GSLIB methods convert to Python 
import os                                               # to set current working directory 
import sys                                              # supress output to screen for interactive variogram modeling
import io
import numpy as np                                      # arrays and matrix math
import pandas as pd                                     # DataFrames
import matplotlib.pyplot as plt                         # plotting
from matplotlib.pyplot import cm                        # color maps
from ipywidgets import interactive                      # widgets and interactivity
from ipywidgets import widgets                            
from ipywidgets import Layout
from ipywidgets import Label
from ipywidgets import VBox, HBox
from matplotlib.patches import Ellipse    
from scipy.optimize import curve_fit              # plot an ellipse
from scipy import stats
fields = ['Name','Length',"X","Y","Z","CU","STRATZ","VEIN"]
pio.renderers.default='browser'
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\Kansanshi_comp_GeolAssay_1p0m.csv", skipinitialspace=True, usecols=fields)
df = df[(pd.to_numeric(df["CU"], errors='coerce')>0)]
df = df.dropna(subset=['CU'])
df['X_'] = 1000
grouped = df.groupby('Name')
df = grouped.filter(lambda x: len(x) >= 20)
#df['CU'] = np.log(df['CU'])
#df['CU'] = stats.zscore(df['CU'])
name = df['Name'].unique()
range_list = []
sill_list = []
nugget_list = []
for i in name[0:1000]:
    df1 = df[df['Name']==i]
    df1 = df1.sort_values('Z', ascending=True)
    xmin = 1000.0; xmax = 1500.0                                # spatial extents in x and y
    ymin = 1000.0; ymax = 1500.0
    feature = 'CU'; feature_units = '%'         # name and units of the feature of interest
    vmin = 0.0; vmax = 22.0                                  # min and max of the feature of interest
    cmap = plt.cm.inferno                                    # set the color map
    # plt.subplot(111)                                        # location map of normal score transform of porosity
    # GSLIB.locmap_st(data,'X_','Z',feature,500,1500,1300,1320,vmin,vmax,feature,'X (m)','Y (m)',feature,cmap)
    # plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.1, wspace=0.5, hspace=0.3)
    # plt.show()
    tmin = -9999.; tmax = 9999.                             # no trimming 
    lag_dist = round((df1['Z'][1:2].values[0] - df1['Z'][0:1].values[0]),2); lag_tol = lag_dist/2; nlag = len(df1);            # maximum lag is 700m and tolerance > 1/2 lag distance for smoothing
    bandh = 9999.9; atol = 0                            # no bandwidth, directional variograms
    isill = 0                                               # standardize sill
    azi_mat = [0]           # directions in azimuth to consider
    
    lag = np.zeros((len(azi_mat),nlag+2)); gamma = np.zeros((len(azi_mat),nlag+2)); npp = np.zeros((len(azi_mat),nlag+2));
    
    for iazi in range(0,len(azi_mat)):                      # Loop over all directions
        lag[iazi,:], gamma[iazi,:], npp[iazi,:] = geostats.gamv(df1,"X_","Z","CU",tmin,tmax,lag_dist,lag_tol,nlag,azi_mat[iazi],atol,bandh,isill)
        #plt.subplot(4,2,iazi+1)
        #plt.plot(lag[iazi,:],gamma[iazi,:],'.',color = 'black',label = 'variogram1 ')
        #plt.plot(variogram.bins,variogram.experimental,'.',color = 'red',label = 'variogram2 ')
        #plt.plot([0,2000],[1.0,1.0],color = 'black')
    #     plt.xlabel(r'Lag Distance $\bf(h)$, (m)')
    #     plt.ylabel(r'$\gamma \bf(h)$')
    #     plt.title('Variogram for single bore core in Kansanshi')
    #     plt.xlim([0,50])
    #     plt.ylim([0,2])
    #     plt.legend(loc='upper left')
    #     plt.grid(True)
    # plt.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=4.2, wspace=0.2, hspace=0.3)
    # plt.show()
    
    
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    import numpy as np
    from skgstat import models
    xdata = lag[iazi,:]
    ydata = gamma[iazi,:]
    p0 = [np.mean(xdata), np.mean(ydata), 0]
    cof, cov =curve_fit(models.spherical, xdata, ydata, p0=p0)
    range_list.append(cof[0])
    sill_list.append(cof[1])
    nugget_list.append(cof[2])
    # print("range: %.2f   sill: %.f   nugget: %.2f" % (cof[0], cof[1], cof[2]))
    # xi = np.linspace(xdata[0], xdata.max(), 100)
    # yi = [models.spherical(h, *cof) for h in xi]
    # plt.plot(xdata, ydata, 'og')
    # plt.plot(xi, yi, '-b');
    # plt.xlabel('lag distance')
    # plt.ylabel('variance')
    # plt.ylim([0,2])
        #plt.plot(df['CU'])
    
    index1 = [i for i,v in enumerate(range_list) if v <0.1]
    filter1 = list(name[index1])
    list_mean1 = []
    list_var1 = []
    for i in filter1:
        df1 = df[df['Name']==i]
        list_mean1.append(df1['CU'].mean())
        list_var1.append(df1['CU'].var())

    df1 = df[df['Name']==filter1[0]]





df1 = df[df['Name']==name[4]]
df1 = df1.sort_values('Z', ascending=True)
xmin = 1000.0; xmax = 1500.0                                # spatial extents in x and y
ymin = 1000.0; ymax = 1500.0
feature = 'CU'; feature_units = '%'         # name and units of the feature of interest
vmin = 0.0; vmax = 22.0                                  # min and max of the feature of interest
cmap = plt.cm.inferno                                    # set the color map
# plt.subplot(111)                                        # location map of normal score transform of porosity
# GSLIB.locmap_st(data,'X_','Z',feature,500,1500,1300,1320,vmin,vmax,feature,'X (m)','Y (m)',feature,cmap)
# plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.1, wspace=0.5, hspace=0.3)
# plt.show()
tmin = -9999.; tmax = 9999.                             # no trimming 
lag_dist = round((df1['Z'][1:2].values[0] - df1['Z'][0:1].values[0]),2); lag_tol = lag_dist/2; nlag = len(df1);            # maximum lag is 700m and tolerance > 1/2 lag distance for smoothing
bandh = 9999.9; atol = 0                            # no bandwidth, directional variograms
isill = 0                                               # standardize sill
azi_mat = [0]           # directions in azimuth to consider

lag = np.zeros((len(azi_mat),nlag+2)); gamma = np.zeros((len(azi_mat),nlag+2)); npp = np.zeros((len(azi_mat),nlag+2));

for iazi in range(0,len(azi_mat)):                      # Loop over all directions
    lag[iazi,:], gamma[iazi,:], npp[iazi,:] = geostats.gamv(df1,"X_","Z","CU",tmin,tmax,lag_dist,lag_tol,nlag,azi_mat[iazi],atol,bandh,isill)

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from skgstat import models
xdata = lag[iazi,:]
ydata = gamma[iazi,:]
p0 = [np.mean(xdata), np.mean(ydata), 0]
cof, cov =curve_fit(models.spherical, xdata, ydata, p0=p0)
print("range: %.2f   sill: %.f   nugget: %.2f" % (cof[0], cof[1], cof[2]))
xi = np.linspace(xdata[0], xdata.max(), 100)
yi = [models.spherical(h, *cof) for h in xi]
plt.plot(xdata, ydata, 'og')
plt.plot(xi, yi, '-b');
plt.xlabel('lag distance')
plt.ylabel('variance')














