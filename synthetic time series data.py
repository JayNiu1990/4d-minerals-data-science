import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
n = 100  # Length of the time series
t = np.arange(0, n)
# Generate white noise with zero mean and unit variance
white_noise = np.random.randn(n)
def generate_stationary_AR1(white_noise, phi):
    n = len(white_noise)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + white_noise[t]
    return x

phi = 0.6  # Autoregressive parameter
stationary_series = generate_stationary_AR1(white_noise, phi)

fig,axis = plt.subplots(1,1,figsize=(12,6),sharey=False,sharex=False);
axis.plot(t, stationary_series,color='b')
axis.set_xlabel('Time',fontsize=18)
axis.set_ylabel('Value',fontsize=18)

axis.set_title('Synthetic Stationary Time Series',fontsize=18)
axis.tick_params(axis='both', which='major', labelsize=18)
fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\1.png', bbox_inches='tight',dpi=300)

n = 100  # Length of the time series
t = np.arange(0, n)
def generate_non_stationary_trend(n, slope=0.1):
    t = np.arange(0, n)
    trend = slope * t
    white_noise = np.random.randn(n)
    non_stationary_series = trend + white_noise
    return non_stationary_series

slope = 0.1  # Trend slope
non_stationary_series = generate_non_stationary_trend(n, slope)
fig,axis = plt.subplots(1,1,figsize=(12,6),sharey=False,sharex=False);
axis.plot(t, non_stationary_series,color='b')
axis.set_xlabel('Time',fontsize=18)
axis.set_ylabel('Value',fontsize=18)

axis.set_title('Synthetic Non-Stationary Time Series',fontsize=18)
axis.tick_params(axis='both', which='major', labelsize=18)
fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\2.png', bbox_inches='tight',dpi=300)