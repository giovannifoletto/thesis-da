# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# Giovanni Foletto
# > This script is the second of a bigger pipeline. It is only responsible of taking input, train and 
# > evaluate the model of the project.

import numpy as np
import pandas as pd
import polars as pl

import torch
import torch.nn as nn
import torchtext as tt
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

import json
from tqdm import tqdm

from collections import Counter
from copy import deepcopy as dc

from sklearn.preprocessing import MinMaxScaler

df_new_data = pl.read_csv("../../data/prepared/output_tesor_data.csv")

df_new_data = df_new_data.with_columns(pl.col("eventTime").str.to_datetime())
df_new_data.head(2)

sns.lineplot(
    y="tokenized_value",
    x="eventTime",
    data=df_new_data.to_pandas()[:5000]
  )

sw = 7 # sliding windows

# this will prepare LSTM for the training and testing
for i in range(1, sw+1):
  df_new_data = df_new_data.with_columns(
    pl.Series(df_new_data["tokenized_value"]).shift(i).alias(f"t-{i}")
  )

df_new_data = df_new_data.drop_nulls()
df_new_data.head(2)

shifted_df_as_np = df_new_data.to_numpy()
shifted_df_as_np

df_to_numpy_to_scale = shifted_df_as_np[:, 3:] # take only the columns with real interesting data

scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(df_to_numpy_to_scale)

shifted_df_as_np[0].shape

X = shifted_df_as_np[:, 1:]
y = shifted_df_as_np[:, 0]

X = dc(np.flip(X, axis=1)) # this function will flip the columns. Now the older value is t-1, the newest is t-7

where_to_split = 0.65

split_index = int(len(X)*where_to_split)
split_index

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

(X_train.shape, X_test.shape, y_train.shape, y_test.shape) # check on dimension

# Reshape since the LSTM in pythorch need one extra dimension

X_train = X_train.reshape((-1, sw, 1))
X_test = X_test.reshape((-1, sw, 1))

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))
X_train.shape, X_test.shape, y_train.shape, y_test.shape # another dimensional check

# Transfrom to pytorch
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()

X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

X_train.shape, X_test.shape, y_train.shape, y_test.shape

class LogSeriesForecastingDataset(Dataset):
  def __init__(self, X, y):
    self.X = X
    self.y = y
  def __len__(self):
    return len(self.X)

  def __getitem__(self, i):
    return self.X[i], self.y[i]

train_dataset = LogSeriesForecastingDataset(X_train, y_train)
test_dataset = LogSeriesForecastingDataset(X_test, y_test)

batch_size = 2048

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device

# check on data loaded
for _, batch in enumerate(train_loader):
  x_batch, y_batch = batch[0].to(device), batch[1].to(device)
  print(x_batch.shape, y_batch.shape)
  break

# check on data loaded
for _, batch in enumerate(test_loader):
  x_batch, y_batch = batch[0].to(device), batch[1].to(device)
  print(x_batch.shape, y_batch.shape)
  break

class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_stacked_layers):
    super().__init__()

    self.hidden_size = hidden_size
    self.num_stacked_layers = num_stacked_layers

    self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, 1)

  def forward(self, x):
    batch_size = x.size(0)
    h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
    c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, 1, :])
    return out

model = LSTM(1, 64, 2)
model.to(device)
model

when_to_see_info = 10000

def train_one_epoch():
  model.train(True)
  print(f"Epoch: {epoch+1}")
  running_loss = 0.0

  for batch_index, batch in tqdm(enumerate(train_loader)):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)

    output = model(x_batch)
    loss = loss_function(output, y_batch)
    running_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_index % when_to_see_info == when_to_see_info -1:
      avg_loss_across_batches = running_loss/when_to_see_info
      print("Batch {0}, Loss {1:.10f}".format(
          batch_index+1,
          avg_loss_across_batches
      ))
      running_loss = 0.0

def validate_one_epoch():
  model.train(False)
  running_loss = 0.0

  for batch_index in enumerate(test_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)

    with torch.no_grad():
      output = model(x_batch)
      loss = loss_function(output, y_batch)
      running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

  print("Val Loss: {0:.10f}".format(avg_loss_across_batches))
  print('***************************************************')

learning_rate = 1e-3
num_epochs = 350
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  train_one_epoch()
  validate_one_epoch()

  print("Saving current model")
  torch.save(model, "drive/MyDrive/model_log_lstm_1.pth")

with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy()

plt.plot(y_train[:500], label='Real Log Template')
plt.plot(predicted[:500], label='Predicted Log Template')
plt.xlabel('Time')
plt.ylabel('Log Template')
plt.legend()
plt.show()
# this is done with LSTM(1, 4, 1)

# create single value representation of the log: DONE
# create timeseries windows: DONE (using sw=7)
# Scaler: DONE
# Split dataset: DONE
# Torch.tensor: DONE
# Dataset/Dataloader w/ batchsize: DONE
# LSTM: DONE
# Save the model

# NOTES:
# - learning_rate too high
# - data too lossy, need some more advanced templating tecnique
# - need more epoch/other parameters

# == From LogDeep ==
# Model
options['input_size'] = 1
options['hidden_size'] = 64
options['num_layers'] = 2
#This options tells that this is multivariate.
options['num_classes'] = 28

# Train
options['batch_size'] = 2048
options['accumulation_step'] = 1

# FROM LogDeep

from torch.autograd import Variable


class deeplog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(deeplog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, features, device):
        input0 = features[0]
        h0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        out, _ = self.lstm(input0, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class loganomaly(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(loganomaly, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm0 = nn.LSTM(input_size,
                             hidden_size,
                             num_layers,
                             batch_first=True)
        self.lstm1 = nn.LSTM(input_size,
                             hidden_size,
                             num_layers,
                             batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, num_keys)
        self.attention_size = self.hidden_size

        self.w_omega = Variable(
            torch.zeros(self.hidden_size, self.attention_size))
        self.u_omega = Variable(torch.zeros(self.attention_size))

        self.sequence_length = 28

    def attention_net(self, lstm_output):
        output_reshape = torch.Tensor.reshape(lstm_output,
                                              [-1, self.hidden_size])
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        attn_hidden_layer = torch.mm(
            attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer),
                                    [-1, self.sequence_length])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas,
                                              [-1, self.sequence_length, 1])
        state = lstm_output
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output

    def forward(self, features, device):
        input0, input1 = features[0], features[1]

        h0_0 = torch.zeros(self.num_layers, input0.size(0),
                           self.hidden_size).to(device)
        c0_0 = torch.zeros(self.num_layers, input0.size(0),
                           self.hidden_size).to(device)

        out0, _ = self.lstm0(input0, (h0_0, c0_0))

        h0_1 = torch.zeros(self.num_layers, input1.size(0),
                           self.hidden_size).to(device)
        c0_1 = torch.zeros(self.num_layers, input1.size(0),
                           self.hidden_size).to(device)

        out1, _ = self.lstm1(input1, (h0_1, c0_1))
        multi_out = torch.cat((out0[:, -1, :], out1[:, -1, :]), -1)
        out = self.fc(multi_out)
        return out


class robustlog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(robustlog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, features, device):
        input0 = features[0]
        h0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        out, _ = self.lstm(input0, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# move data from SKLearn Scaler back to normal data
test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

dummies = np.zeros((X_test.shape[0], sliding_windows+1))
dummies[:, 0] = test_predictions
dummies = scaler.inverse_transform(dummies)

test_predictions = dc(dummies[:, 0])
test_predictions