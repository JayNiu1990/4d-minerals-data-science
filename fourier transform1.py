import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import time
import h5py
from scipy.ndimage.interpolation import rotate
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy import stats
df = pd.read_csv(r'C:\Users\NIU004\OneDrive - CSIRO\Desktop\Data_3.csv')
df = df[['data_value1']]
df1 = df[['data_value1']]
# df1 = df1[['data_value1']]
# df1.loc[df1['data_value1'] < 10] = df1 +10
#df1['data_value1'] = (df1['data_value1'] -df1['data_value1'].mean())/(df1['data_value1'].std())
df1['data_value1'] = np.log(df1['data_value1']+0.01)
#df['data_value1'] = (df1['data_value1'] -df1['data_value1'].mean())/(df1['data_value1'].std())

df1 = df1.reset_index(drop=True)

df1 = np.array(df1).reshape(-1)
dt = 1
t = np.arange(0, len(df1), dt)
ts_ft = np.abs(np.fft.rfft(df1))
ts_freq = np.fft.rfftfreq(len(df1))
plt.figure(figsize=(18, 8))
plt.plot(ts_freq[2:],ts_ft[2:])

n = len(df1)
fhat = np.fft.fft(df1, n) #computes the fft
psd = fhat * np.conj(fhat)/n
freq = (1/(dt*n)) * np.arange(n) #frequency array
idxs_half = np.arange(1, np.floor(n/2), dtype=np.int32) #first half index
## Filter out noise
threshold = 0.5
psd_idxs = psd > threshold #array of 0 and 1
psd_clean = psd * psd_idxs #zero out all the unnecessary powers
fhat_clean = psd_idxs * fhat #used to retrieve the signal

signal_filtered = np.abs(np.fft.ifft(fhat_clean)) #inverse fourier transform
a = pd.DataFrame()
a['value'] = signal_filtered

a.to_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\filter.csv',index=False)
## Visualization
fig, ax = plt.subplots(4,1)
ax[0].plot(t, df1, color='b', lw=0.5, label='Noisy Signal')
ax[0].set_xlabel('t axis')
ax[0].set_ylabel('Vals')
ax[0].set_xlim(0,200)
ax[0].legend()

ax[1].plot(freq[idxs_half], np.abs(psd[idxs_half]), color='b', lw=0.5, label='PSD noisy')
ax[1].set_xlabel('Frequencies in Hz')
ax[1].set_ylabel('Amplitude')
ax[1].legend()
# ax[1].set_xlim(0,200)

ax[2].plot(freq[idxs_half], np.abs(psd_clean[idxs_half]), color='r', lw=1, label='PSD clean')
ax[2].set_xlabel('Frequencies in Hz')
ax[2].set_ylabel('Amplitude')
ax[2].legend()
# ax[2].set_xlim(0,200)

ax[3].plot(t, signal_filtered, color='r', lw=1, label='Clean Signal Retrieved')
ax[3].set_xlabel('t axis')
ax[3].set_ylabel('Vals')
ax[3].legend()
ax[3].set_ylim([0, 3])
# ax[3].set_xlim(0,200)
#plt.subplots_adjust(hspace=0.4)
#plt.savefig('signal-analysis.png', bbox_inches='tight', dpi=300)

plt.plot(signal_filtered,color='b')
plt.plot(df1,alpha=0.5,color='r')
plt.xlim(0,100)








# def ts_smooth(df, threshold=50):
#     fourier = np.fft.rfft(df)
#     frequencies = np.fft.rfftfreq(len(df),d=1)
#     fourier[frequencies > threshold] = 0
#     return np.fft.irfft(fourier,n=len(df1))
# df2 = pd.DataFrame()
# df2['smoothed'] = ts_smooth(df1)
# frequencies = np.fft.rfftfreq(len(df),d=1)

# with plt.style.context('ggplot'):
#     fig, ax = plt.subplots(figsize=(18,8))
# ax.plot(df1,linewidth=2)
# ax.plot(df2, linewidth=3)
# ax.set_title('FFT Smoothing', fontsize=23);
# ax.set_ylabel('g', fontsize=22);
# ax.set_xlabel('time step', fontsize=22);
# #ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
# plt.gcf().autofmt_xdate()
# ax.tick_params(axis='y', labelsize=16)
# ax.tick_params(axis='x', labelsize=16)
# plt.legend(['Real data','FFT smoothing'], fontsize=15)
# plt.xlim(0,400)
# plt.show()





















