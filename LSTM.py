
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 13:44:02 2022

@author: niu004
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#energy_df = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\energydata_complete.csv')
df = pd.read_csv(r'C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\all_grade_over_2000tonnage.csv')
df = df[0:5000]
df = df.groupby(np.arange(len(df))//1).mean()
# # # plt.plot(df['all grade over 2000tonnage'])
# df['moving average 5'] = ' '
# for i in range(5,50000,1):
#     df['moving average 5'][i] = df['all grade over 2000tonnage'][i-5:i].mean()
# df = df[5:15000]
# df = df.drop(['all grade over 2000tonnage'],axis=1)
# df['all grade over 2000tonnage'] = df['moving average 5']
# df = df.drop(['moving average 5'],axis=1)
df['index'] = df.index
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

def create_sliding_window(data, sequence_length, stride=1):
    X_list, y_list = [], []
    for i in range(len(data)):
      if (i + sequence_length) < len(data):
        X_list.append(data.iloc[i:i+sequence_length:stride, :].values)
        y_list.append(data.iloc[i+sequence_length, 0])
    return np.array(X_list), np.array(y_list)

train_split = 0.7
n_train = int(train_split * len(df))
n_test = int(len(df))- n_train
feature_array = df.values
feature_scaler = MinMaxScaler()
feature_scaler.fit(feature_array[:n_train])
target_scaler = MinMaxScaler()
target_scaler.fit(feature_array[:n_train, 0].reshape(-1, 1))

# Transfom on both Training and Test data
scaled_array = pd.DataFrame(feature_scaler.transform(feature_array))

sequence_length = 10
X, y = create_sliding_window(scaled_array, 
                             sequence_length)

X_train = X[:n_train]
y_train = y[:n_train]

X_test = X[n_train:]
y_test = y[n_train:]


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
device = torch.device("cuda:0")
class BayesianLSTM(nn.Module):

    def __init__(self, n_features, output_length, batch_size):

        super(BayesianLSTM, self).__init__()

        self.batch_size = batch_size # user-defined

        self.hidden_size_1 = 128 #128 # number of encoder cells (from paper)
        self.hidden_size_2 = 32#32 # number of decoder cells (from paper)
        self.stacked_layers = 2 # number of (stacked) LSTM layers for each stage

        self.lstm1 = nn.LSTM(n_features, 
                             self.hidden_size_1, 
                             num_layers=self.stacked_layers,
                             batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size_1,
                             self.hidden_size_2,
                             num_layers=self.stacked_layers,
                             batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size_2, output_length)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        hidden = self.init_hidden1(batch_size)
        output, _ = self.lstm1(x, hidden)
        state = self.init_hidden2(batch_size)
        output, state = self.lstm2(output, state)
        output = output[:, -1, :] # take the last decoder cell's outputs
        y_pred = self.fc(output)
        return y_pred
        
    def init_hidden1(self, batch_size):
        hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1)).to(device)
        cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1)).to(device)
        return hidden_state, cell_state
    
    def init_hidden2(self, batch_size):
        hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2)).to(device)
        cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2)).to(device)
        return hidden_state, cell_state
    
    def loss(self, pred, truth):
        return self.loss_fn(pred, truth)

    def predict(self, X):
        return self(torch.tensor(X, dtype=torch.float32)).view(-1).detach().numpy()

n_features = scaled_array.shape[-1]
sequence_length = 10
output_length = 1

batch_size =16
n_epochs =100
learning_rate = 0.0001

bayesian_lstm = BayesianLSTM(n_features=n_features,
                             output_length=output_length,
                             batch_size = batch_size)
bayesian_lstm = bayesian_lstm.to(device)
criterion = torch.nn.MSELoss()
criterion = criterion.to(device)

optimizer = torch.optim.Adam(bayesian_lstm.parameters(), lr=learning_rate)
# optimizer = torch.optim.RMSprop(bayesian_lstm.parameters(), lr=learning_rate)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.9)
loss_train=[]
loss_test=[]
for e in range(1, n_epochs+1):
# for e in range(1, 2):
    bayesian_lstm.train()
    for b in range(0, len(X_train), batch_size):
        features = X_train[b:b+batch_size,:,:]
        target = y_train[b:b+batch_size]    
        X_batch = torch.tensor(features,dtype=torch.float32)    
        y_batch = torch.tensor(target,dtype=torch.float32)
        #X_batch,y_batch = Variable(features),Variable(target)
        
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        output = bayesian_lstm(X_batch)
        loss = criterion(output, y_batch)  
        loss = loss.to(device)
        loss.backward()
        optimizer.step()    
        optimizer.zero_grad() 
        #print(optimizer.param_groups[0]["lr"])
    #scheduler.step()
    #print('epoch', e, 'loss: ', loss.item())
    loss_train.append(loss.item())
    
    bayesian_lstm.eval()
    with torch.no_grad():
        for b in range(0, len(X_test), batch_size):
            features = X_train[b:b+batch_size,:,:]
            target = y_train[b:b+batch_size]    
            X_batch = torch.tensor(features,dtype=torch.float32)    
            y_batch = torch.tensor(target,dtype=torch.float32)
            
            #X_batch,y_batch = Variable(features),Variable(target)
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            output = bayesian_lstm(X_batch)
            loss = criterion(output, y_batch)  
            loss = loss.to(device)
        loss_test.append(loss.item())

            
    if e % 10 == 0:
      print('epoch', e, 'loss: ', loss.item())
offset = sequence_length

def inverse_transform(y):
  return target_scaler.inverse_transform(y.reshape(-1, 1))
X_train = torch.tensor(X_train,dtype=torch.float32)
X_train = X_train.to(device)

training_df = pd.DataFrame()
training_df['index'] = df['index'].iloc[offset:n_train + offset:1] 
X_train = (torch.tensor(X_train, dtype=torch.float32)).detach().to(device)

training_predictions = bayesian_lstm(X_train)

training_df['all grade over 2000tonnage'] = inverse_transform(training_predictions.detach().cpu().numpy())
training_df['source'] = 'Training Prediction'

training_truth_df = pd.DataFrame()
training_truth_df['index'] = training_df['index']
training_truth_df['all grade over 2000tonnage'] = df['all grade over 2000tonnage'].iloc[offset:n_train + offset:1] 
training_truth_df['source'] = 'True Values'


testing_df = pd.DataFrame()
testing_df['index'] = df['index'].iloc[n_train + offset::1] 
X_test = (torch.tensor(X_test, dtype=torch.float32)).detach().to(device)
testing_predictions = bayesian_lstm(X_test)
testing_df['all grade over 2000tonnage'] = inverse_transform(testing_predictions.detach().cpu().numpy())
testing_df['source'] = 'Test Prediction'

testing_truth_df = pd.DataFrame()
testing_truth_df['index'] = testing_df['index']
testing_truth_df['all grade over 2000tonnage'] = df['all grade over 2000tonnage'].iloc[n_train + offset::1] 
testing_truth_df['source'] = 'True Values'
import plotly.io as pio
pio.renderers.default='browser'
evaluation = pd.concat([training_df, 
                        testing_df,
                        training_truth_df,
                        testing_truth_df
                        ], axis=0)
evaluation['MR grade'] = evaluation['all grade over 2000tonnage']
evaluation['Time Step'] = evaluation['index']
fig = px.line(evaluation.loc[evaluation['index'].between(4000, 5000)],
                  x="Time Step",
                  y="MR grade",
                  color="source",
                  title="MR grade measured every 4seconds vs Time Step")
fig.show()      








bayesian_lstm.load_state_dict(torch.load('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\LSTM.pt'))
bayesian_lstm.eval()

import plotly.graph_objects as go
df1 = pd.read_csv(r'C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\all_grade_over_2000tonnage.csv')
df1 = df1[5000:20000]
df1 = df1.groupby(np.arange(len(df1))//1).mean()
# # # plt.plot(df['all grade over 2000tonnage'])

df1['index'] = df1.index

feature_array1 = df1.values
feature_scaler1 = MinMaxScaler()
feature_scaler1.fit(feature_array1)
target_scaler1 = MinMaxScaler()
target_scaler1.fit(feature_array1[:, 0].reshape(-1, 1))

# Transfom on both Training and Test data
scaled_array1 = pd.DataFrame(feature_scaler1.transform(feature_array1))

sequence_length = 10
X1, y1 = create_sliding_window(scaled_array1, 
                             sequence_length)

X_validation = X1
y_validation = y1

X_validation = torch.tensor(X_validation,dtype=torch.float32)
X_validation = X_validation.to(device)

validation_df = pd.DataFrame()
validation_df['index'] = df1['index'].iloc[0 + offset::1] 
X_validation = (torch.tensor(X_validation, dtype=torch.float32)).detach().to(device)

validation_predictions = bayesian_lstm(X_validation)
validation_df['all grade over 2000tonnage'] = inverse_transform(validation_predictions.detach().cpu().numpy())
validation_df['source'] = 'Test Prediction'

validation_truth_df = pd.DataFrame()
validation_truth_df['index'] = validation_df['index']
validation_truth_df['all grade over 2000tonnage'] = df1['all grade over 2000tonnage'].iloc[0 + offset::1] 
validation_truth_df['source'] = 'True Values'

real_trace = go.Scatter(
    x=validation_truth_df['index'],
    y=validation_truth_df['all grade over 2000tonnage'],
    mode='lines',
    fill=None,
    name='Real Values'
    )
predicted_trace_validation = go.Scatter(
    x=validation_df['index'],
    y=validation_df['all grade over 2000tonnage'],
    mode='lines',
    fill=None,
    name='Predicted Values'
    )


data = [real_trace, predicted_trace_validation]
fig = go.Figure(data=data)
fig.update_layout(title='Uncertainty Quantification for MR grade test data measured every 4seconds vs Time Step',
                    xaxis_title='Time',
                    yaxis_title='Copper Grade (wt%)')
fig.show()

# torch.save(bayesian_lstm.state_dict(), 'C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\LSTM.pt')






