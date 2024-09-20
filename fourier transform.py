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
import matplotlib.cm as cmap

#df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\2min_6min_time_elapse_individual_truck_grade_Oct21_Mar22.csv")
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\all_grade_over_2000tonnage.csv")
df1 = df.groupby(np.arange(len(df))//15).mean()
df1 = np.array(df1).reshape(-1)
dt = 1
t = np.arange(0, len(df1), dt)
ts_ft = np.abs(np.fft.rfft(df1))
ts_freq = np.fft.rfftfreq(len(df1))
plt.figure(figsize=(18, 8))
plt.plot(ts_freq[2:],ts_ft[2:])
plt.ylim(0,50)

n = len(df1)
fhat = np.fft.fft(df1, n) #computes the fft
psd = fhat * np.conj(fhat)/n
freq = (1/(dt*n)) * np.arange(n) #frequency array
idxs_half = np.arange(1, np.floor(n/2), dtype=np.int32) #first half index
## Filter out noise
threshold = 0.3
psd_idxs = psd > threshold #array of 0 and 1
psd_clean = psd * psd_idxs #zero out all the unnecessary powers
fhat_clean = psd_idxs * fhat #used to retrieve the signal

signal_filtered = np.abs(np.fft.ifft(fhat_clean)) #inverse fourier transform
a = pd.DataFrame()
a['all grade over 2000tonnage'] = signal_filtered

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
plt.subplots_adjust(hspace=0.4)
plt.savefig('signal-analysis.png', bbox_inches='tight', dpi=300)

plt.plot(signal_filtered)
plt.plot(df1,alpha=0.5,color='r')
plt.xlim(0,200)











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





















