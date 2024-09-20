import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.datasets.mnist as mnist
import time
import torch
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from numpy import genfromtxt
import torch.nn.functional as F
import pandas as pd
import random
import plotly.express as px
import plotly.io as pio
from torch.autograd import Variable
import collections
from sklearn.preprocessing import StandardScaler,MinMaxScaler
pio.renderers.default='browser'
# a = genfromtxt('C:\\Users\\z5011505\\Desktop\\NorthSea_steady_Ca1e-5\\MF_all_TRAIN.txt', delimiter=',')
# b = genfromtxt('C:\\Users\\z5011505\\Desktop\\NorthSea_steady_Ca1e-5\\MF_all_TEST.txt', delimiter=',')
fields = ['BHID', 'X','Y','Z','CuT_dh','Fe_dh','LITH','As_dh']
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Escondida Trucks\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)
df = df.loc[(pd.to_numeric(df["CuT_dh"], errors='coerce')>0.02)
            &(pd.to_numeric(df["X"], errors='coerce')>16000)&(pd.to_numeric(df["X"], errors='coerce')<18000)
            &(pd.to_numeric(df["Y"], errors='coerce')>106000)&(pd.to_numeric(df["Y"], errors='coerce')<108000)]
stdsc = StandardScaler()
df["CuT_dh"] = df["CuT_dh"].astype("float")


# def standardise(series):
#     return (series-series.min())/(series.max()-series.min())

df1 = pd.DataFrame()
df1['X1'] = stdsc.fit_transform(np.array(df['X']).reshape(-1,1)).reshape(-1)
df1['Y1'] = stdsc.fit_transform(np.array(df['Y']).reshape(-1,1)).reshape(-1)
df1['Z1'] = stdsc.fit_transform(np.array(df['Z']).reshape(-1,1)).reshape(-1)
df1['grade'] = np.log(np.array(df['CuT_dh']))

df2 = df1.sample(frac=1)
df2 = df2.reset_index(drop=True)
a = np.array(df2[0:int(0.8*len(df2))]).astype('float64')
b = np.array(df2[int(0.8*len(df2)):]).astype('float64')
# device = torch.device('cuda:0')
device = torch.device('cpu')
time1 = time.time()
class DealDataset1(Dataset):
    def __init__(self):
        self.x_data = torch.from_numpy(a[:, 0:-1])
        self.y_data = torch.from_numpy(a[:, [-1]])
        self.len = a.shape[0]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len
class DealDataset2(Dataset):
    def __init__(self):
        self.x_data = torch.from_numpy(b[:, 0:-1])
        self.y_data = torch.from_numpy(b[:, [-1]])
        self.len = b.shape[0]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len
dealDataset1 = DealDataset1()
train_loader = DataLoader(dataset = dealDataset1, batch_size = 16,shuffle = True)
dealDataset2 = DealDataset2()
test_loader = DataLoader(dataset = dealDataset2, batch_size = 16,shuffle = False)
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden1,n_hidden2,n_output):
        super(Net,self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2,n_output)
    def forward(self,x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.predict(x)
        return x
net = Net(n_feature=3, n_hidden1=16,n_hidden2=19,n_output=1)
net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(),lr = 0.00175)

loss_func = torch.nn.MSELoss()
loss_func = loss_func.to(device)
total_training_loss = []
total_testing_loss = []
for epoch in range(200):
    train_loss = 0.
    train_acc = 0.
    for i, batch in enumerate(train_loader):
        input, output = batch
        # input, output = Variable(input),Variable(output)
        input,output = input.to(device),output.to(device)
        predicted = net(input.float())
        loss = loss_func(predicted,output.float())
        loss = loss.to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(epoch)
    print('Train Loss: {:.6f}'.format(train_loss/len(train_loader)))
    total_training_loss.append(train_loss/len(train_loader))
    net.eval()
    test_loss = 0.
    test_acc = 0.
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input, output = batch
            # input, output = Variable(input),Variable(output)
            input,output = input.to(device), output.to(device)
            predicted = net(input.float())
            loss = loss_func(predicted,output.float())
            loss = loss.to(device)
            test_loss += loss.item()
        print('Test Loss: {:.6f}'.format(test_loss/len(test_loader)))
        total_testing_loss.append(test_loss/len(test_loader))

time2 = time.time()
print(time2-time1)


torch.save(net, 'C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\ml\\neural_network\\model\\model.pkl')
x = list(range(0,200))
plt.plot(x,total_training_loss,'b-',label='train')
plt.plot(x,total_testing_loss,'r-',label='test')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

net = torch.load('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\ml\\neural_network\\model\\model.pkl')
net.eval()
y_pred = []
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        input, output = batch
        input,output = input.to(device), output.to(device)
        predicted = net(input.float())
        predicted = predicted.numpy().flatten()
        y_pred.append(predicted)
y_pred = np.concatenate(y_pred)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(y_pred.reshape(-1,1), b[:,3])
r_squared = model.score(y_pred.reshape(-1,1), b[:,3])

fig,ax = plt.subplots(figsize=(12,6))
ax.scatter(y_pred,b[:,3],color='b')
ax.set_xlabel('predicted copper grade (log)',fontsize=22)
ax.set_ylabel('true copper grade (log)',fontsize=22)
ax.set_title('ANN regression')
z = np.polyfit(y_pred, b[:,3], 1)
y_hat = np.poly1d(z)(y_pred)
ax.plot(y_pred,y_hat,'r--')
text = f'$R^2={r_squared:0.3f}$'
plt.gca().text(0.05,0.95,text,transform=plt.gca().transAxes,fontsize=14,verticalalignment='top')
plt.show()






















