if __name__ == '__main__':
    import warnings
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import random
    import os
    from scipy import stats
    df1 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\2min_6min_time_elapse_individual_truck_grade_Oct21_Mar22.csv")
    
    #df1 = stats.zscore(df1)
    onetruck_list = []
    quartertruck_list = []
    onetruck_mean_list = []
    for i in range(len(df1)):
        onetruck1 = list(df1[i:i+1].values[0])
        onetruck2 = [x for x in onetruck1 if str(x) != 'nan']
        onetruck_list.append(onetruck2)
        onetruck_mean_list.append(np.mean(onetruck2))
        quartertruck = onetruck2[0:round(0.25*len(onetruck2))]
        quartertruck_list.append(quartertruck)
        
    quartertruck_mean_list = []
    for i in range(len(quartertruck_list)):
        quartertruck_mean_list.append(np.mean(quartertruck_list[i]))

    df2 = pd.DataFrame()
    df2['quarter truck mean grade'] = quartertruck_mean_list
    df2['all truck mean grade'] = onetruck_mean_list
    df2['distance'] = df2.index
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


    range_list=[]
    sill_list=[]
    nugget_list=[]
    interval = 60
    truck_eachinterval_list = []
    mean_quartertruck_list = [] ###average partial truck grade in each interval
    mean_alltruck_list = []
    for n in range(0,len(df2)-interval,1):
        df3 = df2[n:n+interval]
        df3['truck series number'] = n
        mean_quartertruck_list.append(df2[n-1+interval:n+interval]['quarter truck mean grade'].values[0])
        mean_alltruck_list.append(df2[n-1+interval:n+interval]['all truck mean grade'].values[0])
        df3['X_'] = 1000
        #df3['quarter truck mean grade log'] = np.log(df3['quarter truck mean grade'])
        xmin = 1000.0; xmax = 1500.0                                # spatial extents in x and y
        ymin = 1000.0; ymax = 1500.0
        feature = 'quarter truck mean grade'; feature_units = '%'         # name and units of the feature of interest
        vmin = 0.0; vmax = 22.0                                  # min and max of the feature of interest
        cmap = plt.cm.inferno                                    # set the color map
        # plt.subplot(111)                                        # location map of normal score transform of porosity
        # GSLIB.locmap_st(data,'X_','Z',feature,500,1500,1300,1320,vmin,vmax,feature,'X (m)','Y (m)',feature,cmap)
        # plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.1, wspace=0.5, hspace=0.3)
        # plt.show()
        tmin = -9999.; tmax = 9999.                             # no trimming 
        lag_dist = 1 ; lag_tol = lag_dist/2 ; nlag = interval; step = 50         # maximum lag is 700m and tolerance > 1/2 lag distance for smoothing
        bandh = 9999.9; atol = 0                            # no bandwidth, directional variograms
        isill = 0                                               # standardize sill
        azi_mat = [0]           # directions in azimuth to consider
        
        lag1 = np.zeros((len(azi_mat),nlag-step+2)); gamma1 = np.zeros((len(azi_mat),nlag-step+2)); npp1 = np.zeros((len(azi_mat),nlag-step+2));
        lag2 = np.zeros((len(azi_mat),nlag+2)); gamma2 = np.zeros((len(azi_mat),nlag+2)); npp2 = np.zeros((len(azi_mat),nlag+2));
        for iazi in range(0,1):                      # Loop over all directions
            lag1[iazi,:], gamma1[iazi,:], npp1[iazi,:] = geostats.gamv(df3,"X_","distance","quarter truck mean grade",tmin,tmax,lag_dist,lag_tol,nlag-step,azi_mat[iazi],atol,bandh,isill)
            lag2[iazi,:], gamma2[iazi,:], npp2[iazi,:] = geostats.gamv(df3,"X_","distance","quarter truck mean grade",tmin,tmax,lag_dist,lag_tol,nlag,azi_mat[iazi],atol,bandh,isill)
            plt.subplot(4,2,iazi+1)
            #plt.plot(lag[iazi,:],gamma[iazi,:],'.',color = 'black',label = 'variogram1 ')
        plt.xlabel(r'Lag Distance $\bf(h)$, (m)')
        plt.ylabel(r'$\gamma \bf(h)$')
        plt.title('Variogram for single bore core in Kansanshi')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=4.2, wspace=0.2, hspace=0.3)
        #plt.show()

        from scipy.optimize import curve_fit
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        import numpy as np
        from skgstat import models
        xdata1 = lag1[iazi,:]
        xdata1 = xdata1[xdata1!=0]
        ydata1 = gamma1[iazi,:]
        ydata1 = ydata1[ydata1!=0]
        ydata1 = np.sqrt(ydata1*2) #std
        
        xdata2 = lag2[iazi,:]
        xdata2 = xdata2[xdata2!=0]
        ydata2 = gamma2[iazi,:]
        ydata2 = ydata2[ydata2!=0]
        ydata2 = np.sqrt(ydata2*2) #std
        p0 = [np.mean(xdata1), np.mean(ydata1), 0]
        cof, cov =curve_fit(models.spherical, xdata1, ydata1, p0=p0)
        print("range: %.2f   sill: %.f   nugget: %.2f" % (cof[0], cof[1], cof[2]))
        xi = np.linspace(xdata1[0], xdata1.max(), 100)
        xii = np.linspace(xdata2[0], xdata2.max(), 100)
        yi = [models.spherical(h, *cof) for h in xi]
        yii = [models.spherical(h, *cof) for h in xii]
        # plt.plot(xdata2, ydata2, 'og')
        # plt.plot(xii, yii, '-b');
        # plt.xlabel('lag distance')
        # plt.ylabel('std')
        #plt.show()
        range_list.append(cof[0])
        sill_list.append(cof[1])
        nugget_list.append(cof[2]) #standard deviation
        df3['std in quarter truck'] = cof[2]
        truck_eachinterval_list.append(df3)
        #plt.ylim([0,1])
results = pd.DataFrame(list(zip(mean_alltruck_list,mean_quartertruck_list, nugget_list)),columns = ['mean all truck grade','mean quarter truck grade','std'])

from scipy.stats import norm
# a = 1
# b = 0.3
# fig, ax = plt.subplots(1, 1)
# x = np.linspace(norm.ppf(0.01,loc=a,scale=b),
#                 norm.ppf(0.99,loc=a,scale=b), 100)
# ax.plot(x, norm.pdf(x,loc=a,scale=b),
#        'r-', lw=5, alpha=0.6, label='norm pdf')
# ax.set_xlim(0,2)

# p = 1 - norm.cdf(0.8,loc = a,scale=b )
cutoff = 1
p_list = []
for i in range(0,len(results),1):
    a = results[i:i+1]['mean quarter truck grade'].values[0]
    b = results[i:i+1]['std'].values[0]
    p = 1 - norm.cdf(cutoff,loc = a,scale=b)
    p_list.append(p)
    
results['p'] = p_list
results['sorting std'] = 0
results['sorting'] = 0
results['sorting gt'] = 0
for i in range(len(results)):
    if results[i:i+1]['mean quarter truck grade'].values[0] >=cutoff:
        results['sorting'][i:i+1] = 1
    else:
        pass    
for i in range(len(results)):
    if results[i:i+1]['mean all truck grade'].values[0] >=cutoff:
        results['sorting gt'][i:i+1] = 1
    else:
        pass      
for i in range(len(results)):
    if results[i:i+1]['p'].values[0]>=0.6:
        results['sorting std'][i:i+1] = 1
    else:
        pass     
    
results['misassigned truck'] = 0
for i in range(len(results)):
    if results['sorting gt'][i:i+1].values[0] != results['sorting'][i:i+1].values[0]:
        results['misassigned truck'][i:i+1] = 1
results['misassigned truck by std'] = 0
for i in range(len(results)):
    if results['sorting gt'][i:i+1].values[0] != results['sorting std'][i:i+1].values[0]:
        results['misassigned truck by std'][i:i+1] = 1        


# a = 1
# b = 0.144
# fig, ax = plt.subplots(1, 1)
# x = np.linspace(norm.ppf(0.01,loc=a,scale=b),
#                 norm.ppf(0.99,loc=a,scale=b), 100)
# ax.plot(x, norm.pdf(x,loc=a,scale=b),
#         'r-', lw=5, alpha=0.6, label='norm pdf')
# ax.set_xlim(0,2)


# plt.hist(results['std'],bins=200)
# plt.xlim(0,1)
# plt.hist(np.log(results['mean quarter truck grade']),bins=200,color='r',density=True)
# plt.hist(np.log(results['mean all truck grade']),bins=200,color='g',density=True)

# count = 1
# for i in range(len(results)):
#     if results['mean quarter truck grade'][i:i+1].values[0] > results['mean all truck grade'][i:i+1].values[0]:
#         count +=1