
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd    
import plotly.io as pio
import plotly.graph_objs as go
from scipy import stats
import random
from sklearn.metrics import mean_squared_error
from scipy.stats import invgamma


import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Gaussian distribution
mean = 0.6  # Mean of the distribution
std_dev = 0.1  # Standard deviation of the distribution

# Generate random data following a Gaussian distribution
data = np.random.normal(mean, std_dev, 1000)


# Plot the probability density function (PDF)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.exp(-0.5*((x - mean)/std_dev)**2) / (std_dev * np.sqrt(2 * np.pi))



fig,axis = plt.subplots(1,1,figsize=(12,8),sharey=False,sharex=False); 
axis.plot(x, p, 'k', linewidth=2)
axis.hist(data, bins=30, density=True, alpha=0.6, color='b')
axis.set_title("Gaussian Distribution\nmean = {}, std_dev = {}".format(mean, std_dev),fontsize=24)
axis.tick_params(axis='both', which='major', labelsize=24)
axis.set_xlim([0, 2])
axis.set_xlabel('Cu grade',fontsize=24)
axis.set_ylabel('Density',fontsize=24)
axis.legend(loc='upper right',fontsize=24)



# Parameters for the Gaussian distribution
mean = 0.6  # Mean of the distribution
std_dev = 0.2  # Standard deviation of the distribution

# Generate random data following a Gaussian distribution
data = np.random.normal(mean, std_dev, 1000)


# Plot the probability density function (PDF)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.exp(-0.5*((x - mean)/std_dev)**2) / (std_dev * np.sqrt(2 * np.pi))



fig,axis = plt.subplots(1,1,figsize=(12,8),sharey=False,sharex=False); 
axis.plot(x, p, 'k', linewidth=2)
axis.hist(data, bins=30, density=True, alpha=0.6, color='b')
axis.set_title("Gaussian Distribution\nmean = {}, std_dev = {}".format(mean, std_dev),fontsize=24)
axis.tick_params(axis='both', which='major', labelsize=24)
axis.set_xlim([0, 2])
axis.set_xlabel('Cu grade',fontsize=24)
axis.set_ylabel('Density',fontsize=24)
axis.legend(loc='upper right',fontsize=24)


mra = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi MRA Time3\\2021-10-10.csv")
fig,axis = plt.subplots(1,1,figsize=(16,5),sharey=False,sharex=False); 
axis.plot(mra.index,mra['tonnage'])
axis.set_xlabel('Time',fontsize=24)
axis.set_ylabel('Tonnage',fontsize=24)


fig,axis = plt.subplots(1,1,figsize=(16,5),sharey=False,sharex=False); 
axis.plot(mra.index,mra['grade'])
axis.set_xlabel('Time',fontsize=24)
axis.set_ylabel('Cu grade (w.t%)',fontsize=24)




















import numpy as np
import matplotlib.pyplot as plt

# Step 1: Set the parameters for the distributions
mu, sigma = 0.2, 0.2  # mean and standard deviation
# Step 2: Generate log-normal distribution data
log_normal_data = np.random.lognormal(mu, sigma, 1000)
# Step 3: Generate normal distribution data
normal_data = np.random.normal(mu, sigma, 1000)


fig,axis = plt.subplots(1,1,figsize=(12,8),sharey=False,sharex=False); 
axis.hist(normal_data, bins=30, density=True, alpha=0.6, color='b')
axis.tick_params(axis='both', which='major', labelsize=24)
axis.set_xlim([-2, 3])
axis.set_xlabel('Log transformed Cu grade',fontsize=24)
axis.set_ylabel('Density',fontsize=24)




import numpy as np
import matplotlib.pyplot as plt

# Step 1: Set the parameters for the distributions
mu, sigma = 0.2, 0.5  # mean and standard deviation
# Step 2: Generate log-normal distribution data
log_normal_data = np.random.lognormal(mu, sigma, 1000)
# Step 3: Generate normal distribution data
normal_data = np.random.normal(mu, sigma, 1000)


fig,axis = plt.subplots(1,1,figsize=(12,8),sharey=False,sharex=False); 
axis.hist(normal_data, bins=30, density=True, alpha=0.6, color='b')
axis.tick_params(axis='both', which='major', labelsize=24)
axis.set_xlim([-2, 3])
axis.set_xlabel('Log transformed Cu grade',fontsize=24)
axis.set_ylabel('Density',fontsize=24)







import numpy as np
import matplotlib.pyplot as plt

# Step 1: Set the parameters for the distributions
mu, sigma = 0.1, 0.5  # mean and standard deviation

# Step 2: Generate log-normal distribution data
log_normal_data = np.random.lognormal(mu, sigma, 1000)

# Step 3: Generate normal distribution data
normal_data = np.random.normal(mu, sigma, 1000)

# Step 4: Plot the distributions
plt.figure(figsize=(10, 5))

# Plot the log-normal distribution
plt.subplot(1, 2, 1)
plt.hist(log_normal_data, bins=30, density=True, alpha=0.6, color='g')
plt.title('Log-Normal Distribution')

# Plot the normal distribution
plt.subplot(1, 2, 2)
plt.hist(normal_data, bins=30, density=True, alpha=0.6, color='b')
plt.title('Normal Distribution')

plt.show()










