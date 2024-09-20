import os
from skimage import io
import torchvision.datasets.mnist as mnist
import time
from torch.autograd import Variable
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import torch.multiprocessing as mp
from multiprocessing import Process
#img = io.imread('C:\\Users\\z5011505\\Desktop\\SLICE\\SLICE2_PNG\\SLICE2_DRY_RESIZE_8bit.png')
#img = cv2.copyMakeBorder(img,40,40,40,40,cv2.BORDER_REPLICATE)
#a = img[40:540,540:1040]
#plt.imshow(a)
#plt.show()
#img = np.array(img)
'''
rootDir = 'C:\\Users\\z5011505\\Desktop\\New folder\\'
img = io.imread('C:\\Users\\z5011505\\Desktop\\SLICE\\SLICE1_PNG\\SLICE1_DRY.png')
f1 = open('C:\\Users\\z5011505\\Desktop\\New folder\\1000x1000.txt','w')
img = cv2.copyMakeBorder(img,40,40,40,40,cv2.BORDER_REPLICATE)
i=1
for y in range(1540,4540,500):
    for x in range(1840,4840,500):
        a = img[y:y+500,x:x+500]
        io.imsave(rootDir + str(i) +'.png',a)
        f1.write(rootDir + str(i) +'.png' +'\n')
        i +=1
f1.close()
'''
rootDir = 'C:\\Users\\z5011505\\Desktop\\3D SLICE\\block00000000_1\\'
f1 = open('C:\\Users\\z5011505\\Desktop\\3D SLICE\\block00000000_1\\1.txt','r')
lines = f1.readlines()
def imagepatch(line,num):
        line = line.strip()
        img = io.imread(line)
        img = cv2.copyMakeBorder(img,40,40,40,40,cv2.BORDER_REPLICATE)
        imtest = np.ndarray(shape=(81,81))
        img= np.array(img)
        list_of_imtest = []
        list_of_pixel = []
        f = open(rootDir+ str(num) + '\\' + 'train.txt', 'w')
        for yy in range(40,540,1):
            for xx in range(40,540,1):
                imtest = img[yy-40:yy+40+1,xx-40:xx+40+1]
                list_of_imtest.append(imtest)
                pixel = img[yy,xx]
                list_of_pixel.append(pixel)
        count = 0
        for imgpatch, label in zip(list_of_imtest,list_of_pixel):
            img_path= rootDir+ str(num) +'\\' + str(count)+ '.png'
            io.imsave(img_path, imgpatch)
            f.write(img_path + '\n')
            count += 1
        f.close()
if __name__ == '__main__':
    for num,line in enumerate(lines):
        procs = []
        proc = Process(target=imagepatch, args=(line,num,))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
