# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:05:17 2022

@author: niu004
"""

from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
df = pd.read_csv(r'C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\all_grade_over_2000tonnage.csv')
df = df.groupby(np.arange(len(df))//1).mean()
df['index'] = df.index
df = df[0:20000]

def create_sliding_window(data, sequence_length, stride=1):
    X_list, y_list = [], []
    for i in range(len(data)):
      if (i + sequence_length) < len(data):
        X_list.append(data.iloc[i:i+sequence_length:stride, :].values)
        y_list.append(data.iloc[i+sequence_length, 0])
    return np.array(X_list), np.array(y_list)


class BayesianLSTM(nn.Module):

    def __init__(self, n_features, output_length, batch_size,hidden_size1,hidden_size2,stacked_layers_num):

        super(BayesianLSTM, self).__init__()

        self.batch_size = batch_size # user-defined

        self.hidden_size_1 = hidden_size1 #128 # number of encoder cells (from paper)
        self.hidden_size_2 = hidden_size2#32 # number of decoder cells (from paper)
        self.stacked_layers = stacked_layers_num # number of (stacked) LSTM layers for each stage
        self.dropout_probability = 0.5 # arbitrary value (the paper suggests that performance is generally stable across all ranges)

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
        output = F.dropout(output, p=self.dropout_probability, training=True)
        state = self.init_hidden2(batch_size)
        output, state = self.lstm2(output, state)
        output = F.dropout(output, p=self.dropout_probability, training=True)
        output = output[:, -1, :] # take the last decoder cell's outputs
        y_pred = self.fc(output)
        return y_pred
        
    def init_hidden1(self, batch_size):
        hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1))
        cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1))
        return hidden_state, cell_state
    
    def init_hidden2(self, batch_size):
        hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2))
        cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2))
        return hidden_state, cell_state
    
    def loss(self, pred, truth):
        return self.loss_fn(pred, truth)

    def predict(self, X):
        return self(torch.tensor(X, dtype=torch.float32)).view(-1).detach().numpy()

def train(config,epochs = 10, device="cpu"):
    train_split = 0.7
    n_train = int(train_split * len(df))
    n_test = int(len(df))- n_train
    feature_array = df.values

    feature_scaler = MinMaxScaler()
    feature_scaler.fit(feature_array[:n_train])
    
    target_scaler = MinMaxScaler()
    target_scaler.fit(feature_array[:n_train, 0].reshape(-1, 1))
    
    scaled_array = pd.DataFrame(feature_scaler.transform(feature_array))
    
    sequence_length = 10
    X, y = create_sliding_window(scaled_array, 
                                 sequence_length)
    X_train = X[:n_train]
    y_train = y[:n_train]
    
    X_test = X[n_train:]
    y_test = y[n_train:]
    
    n_features = scaled_array.shape[-1]
    sequence_length = 10
    output_length = 1
    batch_size = int(config["batch_size"])
    n_epochs =20
    learning_rate = 0.01
    bayesian_lstm = BayesianLSTM(n_features=n_features,
                                 output_length=output_length,
                                 batch_size = batch_size,
                                 hidden_size1 = config['hidden_size1'],
                                 hidden_size2 = config['hidden_size2'],
                                 stacked_layers_num = config['stacked_layers_num'])
    bayesian_lstm = bayesian_lstm.to(device)
    criterion = torch.nn.MSELoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(bayesian_lstm.parameters(), lr=config["lr"])
    loss_train=[]
    loss_test=[]
    for e in range(1, n_epochs+1):
        bayesian_lstm.train()
        for b in range(0, len(X_train), batch_size):
            features = X_train[b:b+batch_size,:,:]
            target = y_train[b:b+batch_size]     
            X_batch = torch.tensor(features,dtype=torch.float32)    
            y_batch = torch.tensor(target,dtype=torch.float32)
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            output = bayesian_lstm(X_batch)
            train_loss = criterion(output, y_batch)  
            train_loss = train_loss.to(device)
            
            train_loss.backward()
            optimizer.step()    
            optimizer.zero_grad() 

        loss_train.append(train_loss.item())
        val_loss = 0
        val_steps = 0
        bayesian_lstm.eval()
        with torch.no_grad():
            for b in range(0, len(X_test), batch_size):
                features = X_train[b:b+batch_size,:,:]
                target = y_train[b:b+batch_size]    
                X_batch = torch.tensor(features,dtype=torch.float32)    
                y_batch = torch.tensor(target,dtype=torch.float32)
                
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                output = bayesian_lstm(X_batch)
                test_loss = criterion(output, y_batch)  
                test_loss = test_loss.to(device)
                val_loss+=test_loss.cpu().numpy()
                val_steps+=1
            loss_test.append(test_loss.item())
        tune.report(loss=(val_loss / val_steps))       
        if e % 10 == 0:
          print('epoch', e, 'loss: ', train_loss.item())
      
def main(num_samples=50,max_num_epochs=10,gpus_per_trial=0):
    device = torch.device("cuda:0")
    config = {'lr': tune.choice([1e-5,1e-4,1e-3,1e-2]),
              'batch_size': tune.choice([32,64,128]),
              'hidden_size1': tune.choice([8,16,32,64,128]),
              'hidden_size2': tune.choice([8,16,32]),
              'stacked_layers_num': tune.choice([2,3,4])}
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(metric_columns=["loss"])
    result = tune.run(
        partial(train),
        resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)
    best_trial = result.get_best_trial("loss","min")
    if torch.cuda.is_available():
        device = "cuda:0"
        print('using GPU')
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

if __name__ == "__main__":
    main(num_samples=50,max_num_epochs=30,gpus_per_trial=1)
    