import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
#energy_df = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\energydata_complete.csv')
df = pd.read_csv(r'C:\Users\NIU004\OneDrive - CSIRO\Desktop\Data_3.csv')
df['data_value1'] = (df['data_value1'] -df['data_value1'].mean())/(df['data_value1'].std())
df = df[['data_value1']]

# df['data_value1'] = np.log(df['data_value1'] + 0.1)
df['index'] = range(1, len(df) + 1)
df = df[0:10000]
#df = df[['data_value1','data_value2']]
# result = adfuller(df.values)
# print('1. ADF:',result[0])
# print('2. p-value:',result[1])
# print('3. Num of Lags:',result[2])
# print('4. Num of Observations Used For ADF Regression and Critical Values Calculation:', result[3])
# print('5. Critical Values:')
# for key,val in result[4].items():
#     print('\t',key,':',val)
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

train_split = 0.8
n_train = int(train_split * len(df))
n_test = int(len(df))- n_train
feature_array = df.values
# feature_scaler = MinMaxScaler()
# feature_scaler.fit(feature_array[:n_train])
# target_scaler = MinMaxScaler()
# target_scaler.fit(feature_array[:n_train, 0].reshape(-1, 1))

# Transfom on both Training and Test data
scaled_array = pd.DataFrame(feature_array)

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
output_length = 1
batch_size = 32
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
      
torch.save(bayesian_lstm.state_dict(), 'C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Bayesian_LSTM.pt')




#################load model###########################
bayesian_lstm.load_state_dict(torch.load('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Bayesian_LSTM.pt'))
bayesian_lstm.eval()
offset = sequence_length

X_train = torch.tensor(X_train,dtype=torch.float32)
X_train = X_train.to(device)

training_df = pd.DataFrame()
training_df['index'] = df['index'].iloc[offset:n_train + offset:1] 
X_train = (torch.tensor(X_train, dtype=torch.float32)).detach().to(device)

training_predictions = bayesian_lstm(X_train)

training_df['data_value1'] = training_predictions.detach().cpu().numpy()
training_df['source'] = 'Training Prediction'

training_truth_df = pd.DataFrame()
training_truth_df['index'] = training_df['index']
training_truth_df['data_value1'] = df['data_value1'].iloc[offset:n_train + offset:1] 
training_truth_df['source'] = 'True Values'


testing_df = pd.DataFrame()
testing_df['index'] = df['index'].iloc[n_train + offset::1] 
X_test = (torch.tensor(X_test, dtype=torch.float32)).detach().to(device)
testing_predictions = bayesian_lstm(X_test)
testing_df['data_value1'] = testing_predictions.detach().cpu().numpy()
testing_df['source'] = 'Test Prediction'

testing_truth_df = pd.DataFrame()
testing_truth_df['index'] = testing_df['index']
testing_truth_df['data_value1'] = df['data_value1'].iloc[n_train + offset::1] 
testing_truth_df['source'] = 'True Values'
import plotly.io as pio

# evaluation = pd.concat([training_df, 
#                         testing_df,
#                         training_truth_df,
#                         testing_truth_df
#                         ], axis=0)
# evaluation['MR grade'] = evaluation['all grade over 2000tonnage']
# evaluation['Time Step'] = evaluation['index']
# fig = px.line(evaluation.loc[evaluation['index'].between(7000, 8000)],
#                   x="Time Step",
#                   y="MR grade",
#                   color="source",
#                   title="MR grade measured every 4seconds vs Time Step")
# #fig.show()

plt.plot(loss_train,color='r')
plt.plot(loss_test,color='b')


n_experiments = 300
pio.renderers.default='browser'
test_uncertainty_df = pd.DataFrame()
test_uncertainty_df['index'] = testing_df['index']

for i in range(n_experiments):
    X_test = (torch.tensor(X_test, dtype=torch.float32)).detach().to(device)
    experiment_predictions = bayesian_lstm(X_test)
    test_uncertainty_df['grade_{}'.format(i)] = experiment_predictions.detach().cpu().numpy()

grade_df = test_uncertainty_df.filter(like='grade', axis=1)
test_uncertainty_df['grade_mean'] = grade_df.mean(axis=1)
test_uncertainty_df['grade_std'] = grade_df.std(axis=1)

test_uncertainty_df = test_uncertainty_df[['index', 'grade_mean', 'grade_std']]
test_uncertainty_df['lower_bound'] = test_uncertainty_df['grade_mean'] - 3*test_uncertainty_df['grade_std']
test_uncertainty_df['upper_bound'] = test_uncertainty_df['grade_mean'] + 3*test_uncertainty_df['grade_std']

import plotly.graph_objects as go

test_uncertainty_plot_df = test_uncertainty_df.copy(deep=True)
test_uncertainty_plot_df = test_uncertainty_plot_df.loc[test_uncertainty_plot_df['index'].between(len(df)*0.8, len(df)*0.8+2000)]
truth_uncertainty_plot_df = testing_truth_df.copy(deep=True)
truth_uncertainty_plot_df = truth_uncertainty_plot_df.loc[testing_truth_df['index'].between(len(df)*0.8, len(df)*0.8+2000)]

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
    y=truth_uncertainty_plot_df['data_value1'],
    mode='lines',
    fill=None,
    name='Real Values'
    )
predicted_mean_trace_test = go.Scatter(
    x=test_uncertainty_plot_df['index'],
    y=test_uncertainty_plot_df['grade_mean'],
    mode='lines',
    fill=None,
    name='Predicted Mean Values'
    )

data = [upper_trace, lower_trace, real_trace,predicted_mean_trace_test]

fig = go.Figure(data=data)
fig.update_layout(title='Uncertainty Quantification',
                    xaxis_title='Time',
                    yaxis_title='value')

fig.show()










###############validation##################
# df_new = pd.read_csv(r'C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\grade_over_2000tonnage_each_row\PLC_CMP_Data_For11D10M2021Y.csv')

# df_new = df_new.groupby(np.arange(len(df_new))//1).mean()
# # # # # plt.plot(df['all grade over 2000tonnage'])
# # df['moving average 5'] = ' '
# # for i in range(5,50000,1):
# #     df['moving average 5'][i] = df['all grade over 2000tonnage'][i-5:i].mean()
    
# # df1=df[15000:20000]
# # df1 = df1.drop(['all grade over 2000tonnage'],axis=1)
# # df1['all grade over 2000tonnage'] = df1['moving average 5']
# # df1 = df1.drop(['moving average 5'],axis=1)
# df_new['index'] = df_new.index

# feature_array1 = df_new.values
# feature_scaler1 = MinMaxScaler()
# feature_scaler1.fit(feature_array1)
# target_scaler1 = MinMaxScaler()
# target_scaler1.fit(feature_array1[:, 0].reshape(-1, 1))

# # # Transfom on both Training and Test data
# scaled_array1 = pd.DataFrame(feature_scaler1.transform(feature_array1))

# X1, y1 = create_sliding_window(scaled_array1, 
#                               sequence_length)

# X_validation = X1
# y_validation = y1

# X_validation = torch.tensor(X_validation,dtype=torch.float32)
# X_validation = X_validation.to(device)

# validation_df = pd.DataFrame()
# validation_df['index'] = df_new['index'].iloc[0 + offset::1] 
# X_validation = (torch.tensor(X_validation, dtype=torch.float32)).detach().to(device)

# validation_predictions = bayesian_lstm(X_validation)
# validation_df['all grade over 2000tonnage'] = inverse_transform(validation_predictions.detach().cpu().numpy())
# validation_df['source'] = 'Test Prediction'


# n_experiments = 200

# validation_uncertainty_df = pd.DataFrame()
# validation_uncertainty_df['index'] = validation_df['index']

# for i in range(n_experiments):
#     X_validation = (torch.tensor(X_validation, dtype=torch.float32)).detach().to(device)
#     experiment_predictions1 = bayesian_lstm(X_validation)
#     validation_uncertainty_df['grade_{}'.format(i)] = inverse_transform(experiment_predictions1.detach().cpu().numpy())

# grade_df = validation_uncertainty_df.filter(like='grade', axis=1)
# validation_uncertainty_df['grade_mean'] = grade_df.mean(axis=1)
# validation_uncertainty_df['grade_std'] = grade_df.std(axis=1)

# validation_uncertainty_df = validation_uncertainty_df[['index', 'grade_mean', 'grade_std']]
# validation_uncertainty_df['lower_bound'] = validation_uncertainty_df['grade_mean'] - 1*validation_uncertainty_df['grade_std']
# validation_uncertainty_df['upper_bound'] = validation_uncertainty_df['grade_mean'] + 1*validation_uncertainty_df['grade_std']

# import plotly.graph_objects as go
# validation_truth_df = pd.DataFrame()
# validation_truth_df['index'] = validation_df['index']
# validation_truth_df['all grade over 2000tonnage'] = df_new['grade over 2000tonnage '].iloc[0 + offset::1] 
# validation_truth_df['source'] = 'True Values'

# validation_uncertainty_plot_df = validation_uncertainty_df.copy(deep=True)
# validation_uncertainty_plot_df = validation_uncertainty_plot_df.loc[validation_uncertainty_plot_df['index'].between(0, 5000)]
# validation_truth_uncertainty_plot_df = validation_truth_df.copy(deep=True)
# validation_truth_uncertainty_plot_df = validation_truth_uncertainty_plot_df.loc[validation_truth_df['index'].between(0, 5000)]

# upper_trace = go.Scatter(
#     x=validation_uncertainty_plot_df['index'],
#     y=validation_uncertainty_plot_df['upper_bound'],
#     mode='lines',
#     fill=None,
#     name='99% Upper Confidence Bound'
#     )
# lower_trace = go.Scatter(
#     x=validation_uncertainty_plot_df['index'],
#     y=validation_uncertainty_plot_df['lower_bound'],
#     mode='lines',
#     fill='tonexty',
#     fillcolor='rgba(255, 211, 0, 0.1)',
#     name='99% Lower Confidence Bound'
#     )
# real_trace = go.Scatter(
#     x=validation_truth_uncertainty_plot_df['index'],
#     y=validation_truth_uncertainty_plot_df['all grade over 2000tonnage'],
#     mode='lines',
#     fill=None,
#     name='Real Values'
#     )
# predicted_mean_trace_validation = go.Scatter(
#     x=validation_uncertainty_plot_df['index'],
#     y=validation_uncertainty_plot_df['grade_mean'],
#     mode='lines',
#     fill=None,
#     name='Real Values'
#     )


# data = [upper_trace, lower_trace, real_trace, predicted_mean_trace_validation]
# fig = go.Figure(data=data)
# fig.update_layout(title='Uncertainty Quantification for MR grade test data measured every 4seconds vs Time Step',
#                     xaxis_title='Time',
#                     yaxis_title='Copper Grade (wt%)')
# fig.show()



