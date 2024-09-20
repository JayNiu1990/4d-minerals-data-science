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

df = df.groupby(np.arange(len(df))//1).mean()
# # # plt.plot(df['all grade over 2000tonnage'])
df['index'] = df.index
df = df[0:5000]

# df = pd.read_csv(r'C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\filter.csv')
# df['index'] = df.index
# df = df[0:5000]
# # day_of_week=0 corresponds to Monday
# energy_df['day_of_week'] = energy_df['date'].dt.dayofweek.astype(int)
#energy_df['hour_of_day'] = energy_df['date'].dt.hour.astype(int)

#selected_columns = ['date','hour_of_day', 'Appliances']
#energy_df = energy_df[selected_columns]
import numpy as np

#resample_df = energy_df.set_index('date').resample('1H').mean()
#resample_df['date'] = resample_df.index
#resample_df['log_energy_consumption'] = np.log(resample_df['Appliances'])

#datetime_columns = ['date','hour_of_day']
#target_column = 'log_energy_consumption'

#feature_columns = datetime_columns + ['log_energy_consumption']

# For clarity in visualization and presentation, 
# only consider the first 150 hours of data.
#resample_df = resample_df[feature_columns]
import plotly.express as px

#plot_length = 150
#plot_df = resample_df.copy(deep=True).iloc[:plot_length]
#plot_df['weekday'] = plot_df['date'].dt.day_name()

# fig = px.line(plot_df,
#               x="date",
#               y="log_energy_consumption", 
#               color="weekday", 
#               title="Log of Appliance Energy Consumption vs Time")
# fig.show()

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

#features = ['hour_of_day', 'log_energy_consumption']
feature_array = df.values

# Fit Scaler only on Training features
feature_scaler = MinMaxScaler()
feature_scaler.fit(feature_array[:n_train])

# Fit Scaler only on Training target values
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

n_features = scaled_array.shape[-1]
sequence_length = 10
output_length = 1

batch_size = 64
n_epochs =150
learning_rate = 0.0001

bayesian_lstm = BayesianLSTM(n_features=n_features,
                             output_length=output_length,
                             batch_size = batch_size)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(bayesian_lstm.parameters(), lr=learning_rate)
# optimizer = torch.optim.RMSprop(bayesian_lstm.parameters(), lr=learning_rate)
bayesian_lstm.train()
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.9)
loss_train=[]
loss_test=[]
for e in range(1, n_epochs+1):
# for e in range(1, 2):
    for b in range(0, len(X_train), batch_size):
        features = X_train[b:b+batch_size,:,:]
        target = y_train[b:b+batch_size]    
        optimizer.step()  
        X_batch = torch.tensor(features,dtype=torch.float32)    
        y_batch = torch.tensor(target,dtype=torch.float32)

        output = bayesian_lstm(X_batch)
        loss = criterion(output, y_batch)  

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
            output = bayesian_lstm(X_batch)
            loss = criterion(output, y_batch)  
        loss_test.append(loss.item())

            
    if e % 10 == 0:
      print('epoch', e, 'loss: ', loss.item())
      

offset = sequence_length

def inverse_transform(y):
  return target_scaler.inverse_transform(y.reshape(-1, 1))

training_df = pd.DataFrame()
training_df['index'] = df['index'].iloc[offset:n_train + offset:1] 
training_predictions = bayesian_lstm.predict(X_train)
training_df['all grade over 2000tonnage'] = inverse_transform(training_predictions)
training_df['source'] = 'Training Prediction'

training_truth_df = pd.DataFrame()
training_truth_df['index'] = training_df['index']
training_truth_df['all grade over 2000tonnage'] = df['all grade over 2000tonnage'].iloc[offset:n_train + offset:1] 
training_truth_df['source'] = 'True Values'


testing_df = pd.DataFrame()
testing_df['index'] = df['index'].iloc[n_train + offset::1] 
testing_predictions = bayesian_lstm.predict(X_test)
testing_df['all grade over 2000tonnage'] = inverse_transform(testing_predictions)
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

plt.plot(loss_train,color='r')
plt.plot(loss_test,color='b')
# testing_df = pd.DataFrame()
# testing_df['index'] = df['index'].iloc[2000 + offset:12500+offset:1] 
# testing_predictions = bayesian_lstm.predict(X_test)
# testing_df['all grade over 2000tonnage'] = inverse_transform(testing_predictions)
# testing_df['source'] = 'Test Prediction'

# testing_truth_df = pd.DataFrame()
# testing_truth_df['index'] = testing_df['index']
# testing_truth_df['all grade over 2000tonnage'] = df['all grade over 2000tonnage'].iloc[2000 + offset:12500+offset:1] 
# testing_truth_df['source'] = 'True Values'

# evaluation = pd.concat([testing_df,
#                         testing_truth_df
#                         ], axis=0)
# evaluation['MR grade'] = evaluation['all grade over 2000tonnage']
# evaluation['Time Step'] = evaluation['index']
# fig = px.line(evaluation.loc[evaluation['index'].between(2150, 2500)],
#                   x="Time Step",
#                   y="MR grade",
#                   color="source",
#                   title="MR grade measured every 4seconds vs Time Step")
# fig.show()
# n_experiments = 200

# test_uncertainty_df = pd.DataFrame()
# test_uncertainty_df['index'] = testing_df['index']

# for i in range(n_experiments):
#   experiment_predictions = bayesian_lstm.predict(X_test)
#   test_uncertainty_df['grade_{}'.format(i)] = inverse_transform(experiment_predictions)

# grade_df = test_uncertainty_df.filter(like='grade', axis=1)
# test_uncertainty_df['grade_mean'] = grade_df.mean(axis=1)
# test_uncertainty_df['grade_std'] = grade_df.std(axis=1)

# test_uncertainty_df = test_uncertainty_df[['index', 'grade_mean', 'grade_std']]
# test_uncertainty_df['lower_bound'] = test_uncertainty_df['grade_mean'] - 1*test_uncertainty_df['grade_std']
# test_uncertainty_df['upper_bound'] = test_uncertainty_df['grade_mean'] + 1*test_uncertainty_df['grade_std']

# import plotly.graph_objects as go

# test_uncertainty_plot_df = test_uncertainty_df.copy(deep=True)
# test_uncertainty_plot_df = test_uncertainty_plot_df.loc[test_uncertainty_plot_df['index'].between(2100, 2500)]
# truth_uncertainty_plot_df = testing_truth_df.copy(deep=True)
# truth_uncertainty_plot_df = truth_uncertainty_plot_df.loc[testing_truth_df['index'].between(2100, 2500)]

# upper_trace = go.Scatter(
#     x=test_uncertainty_plot_df['index'],
#     y=test_uncertainty_plot_df['upper_bound'],
#     mode='lines',
#     fill=None,
#     name='68% Upper Confidence Bound'
#     )
# lower_trace = go.Scatter(
#     x=test_uncertainty_plot_df['index'],
#     y=test_uncertainty_plot_df['lower_bound'],
#     mode='lines',
#     fill='tonexty',
#     fillcolor='rgba(255, 211, 0, 0.1)',
#     name='68% Lower Confidence Bound'
#     )
# real_trace = go.Scatter(
#     x=truth_uncertainty_plot_df['index'],
#     y=truth_uncertainty_plot_df['all grade over 2000tonnage'],
#     mode='lines',
#     fill=None,
#     name='Real Values'
#     )

# data = [upper_trace, lower_trace, real_trace]

# fig = go.Figure(data=data)
# fig.update_layout(title='Uncertainty Quantification for MR grade test data measured every 4seconds vs Time Step',
#                     xaxis_title='Time',
#                     yaxis_title='Copper Grade (wt%)')

# fig.show()
n_experiments = 300

test_uncertainty_df = pd.DataFrame()
test_uncertainty_df['index'] = testing_df['index']

for i in range(n_experiments):
  experiment_predictions = bayesian_lstm.predict(X_test)
  test_uncertainty_df['grade_{}'.format(i)] = inverse_transform(experiment_predictions)

grade_df = test_uncertainty_df.filter(like='grade', axis=1)
test_uncertainty_df['grade_mean'] = grade_df.mean(axis=1)
test_uncertainty_df['grade_std'] = grade_df.std(axis=1)

test_uncertainty_df = test_uncertainty_df[['index', 'grade_mean', 'grade_std']]
test_uncertainty_df['lower_bound'] = test_uncertainty_df['grade_mean'] - 2*test_uncertainty_df['grade_std']
test_uncertainty_df['upper_bound'] = test_uncertainty_df['grade_mean'] + 2*test_uncertainty_df['grade_std']

import plotly.graph_objects as go

test_uncertainty_plot_df = test_uncertainty_df.copy(deep=True)
test_uncertainty_plot_df = test_uncertainty_plot_df.loc[test_uncertainty_plot_df['index'].between(4000, 5000)]
truth_uncertainty_plot_df = testing_truth_df.copy(deep=True)
truth_uncertainty_plot_df = truth_uncertainty_plot_df.loc[testing_truth_df['index'].between(4000, 5000)]

upper_trace = go.Scatter(
    x=test_uncertainty_plot_df['index'],
    y=test_uncertainty_plot_df['upper_bound'],
    mode='lines',
    fill=None,
    name='99% Upper Confidence Bound'
    )
lower_trace = go.Scatter(
    x=test_uncertainty_plot_df['index'],
    y=test_uncertainty_plot_df['lower_bound'],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(255, 211, 0, 0.1)',
    name='99% Lower Confidence Bound'
    )
real_trace = go.Scatter(
    x=truth_uncertainty_plot_df['index'],
    y=truth_uncertainty_plot_df['all grade over 2000tonnage'],
    mode='lines',
    fill=None,
    name='Real Values'
    )

data = [upper_trace, lower_trace, real_trace]

fig = go.Figure(data=data)
fig.update_layout(title='Uncertainty Quantification for MR grade test data measured every 4seconds vs Time Step',
                    xaxis_title='Time',
                    yaxis_title='Copper Grade (wt%)')

fig.show()











