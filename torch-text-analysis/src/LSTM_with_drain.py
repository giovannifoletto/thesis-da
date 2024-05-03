#!/usr/bin/env python
# coding: utf-8

# Giovanni Foletto, from IPYNB with the same name

import polars as pl
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import gc
from copy import deepcopy as dc
import datetime

input_dir = '../../data/raw' # The input directory of log file
output_dir = '../../data/prepared/drain_parser/'  # The output directory of parsing results
log_file = 'unificated.ndjson'  # The input log file name
log_format = '<Content>' # Define log format to split message fields
# Regular expression list for optional preprocessing (default: [])
regex = [
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)' # IP
]
st = 0.5  # Similarity threshold
depth = 4  # Depth of all leaf nodes


# #####
# Neural Network Settings
# #####

sw = 7 # Sliding windows
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} as device")

where_to_split = 0.90

batch_size = 2048

learning_rate = 1e-3
num_epochs = 300

loss_function = nn.MSELoss()

# #####
# END of configurations
# #####

dlp = pl.read_csv(output_dir + "unificated.ndjson_structured.csv")

print("=== SEE HEAD OF DATASET ===")
print(dlp.head(2))

print("=== CHECK IF SOMETHING IS NULL ===")
print("That means that some errors in the drain templater happened.")
print(f"Number of errors: {dlp.filter(pl.col('EventId').is_null()).shape}")

print("=== Converting to Numpy Array ===")
dlp_np = dlp.select(pl.col("LineId", "EventId")).to_numpy()
print(f"Shape of the numpy array: {dlp_np.shape}")

print("=== import the template dataset ===")
ord_dict = dlp.select(pl.col("EventId")).unique().to_numpy()
print(f"The template dataset has: {ord_dict.shape} (shape)")

new_df_support_array = []

for index, elem in enumerate(ord_dict):
  new_df_support_array.append((index+1, elem.item()))

token_mapping = pl.DataFrame(new_df_support_array)
token_mapping = token_mapping.rename({
    "column_0": "value",
    "column_1": "EventId"
})

print("=== Token Mapping ===")
print(token_mapping)

print("=== Saving to token mapping file ===")
token_mapping.write_csv(output_dir + "token_mapping.csv")

del new_df_support_array
del dlp_np

gc.collect()

#token_mapping = pl.read_csv(output_dir + "token_mapping.csv")

print("=== join on token mapping and original dataset ===")
dlp = dlp.select(pl.col("LineId", "EventId")).join(token_mapping, on="EventId")
print(f"Shape of the final dataset version: {dlp.shape}")

for i in range(1, sw+1):
  dlp = dlp.with_columns(
      pl.Series(dlp["value"]).shift(i).alias(f"t-{i}")
  )
dlp = dlp.drop_nulls().select(pl.col("*").exclude("EventId"))
print(f"Final dataset with {sw} epochs:")
print(dlp.head(2))

#sns.lineplot(data=dlp[:200].to_pandas(), y="value", x="LineId")

print("=== Shifting order ===")
shifted_np = dlp.to_numpy()

print("=== Features scaling ===")
scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_np = scaler.fit_transform(shifted_np)

X = shifted_np[:, 2:]
y = shifted_np[:, 1]

print(f"Dataset shape: (X.shape), (y.shape) {X.shape}, {y.shape}")

X = dc(np.flip(X, axis=1))

split_index = int(len(X)*where_to_split)
print(f"Splitting with this dimensions: {split_index}")

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print(f"(X_train.shape), (X_test.shape), (y_train.shape), (y_test.shape): {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}")

X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()

X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

print(f"(X_train.shape), (X_test.shape), (y_train.shape), (y_test.shape): {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}")

X_train = X_train.reshape((-1, sw, 1))
X_test = X_test.reshape((-1, sw, 1))

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

print(f"(X_train.shape), (X_test.shape), (y_train.shape), (y_test.shape): {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}")

class FlawsLogDataset(Dataset):
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, i):
    return self.X[i], self.y[i]

train_dataset = FlawsLogDataset(X_train, y_train)
test_dataset = FlawsLogDataset(X_test, y_test)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for _, batch in enumerate(train_loader):
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
print("=== MODEL CREATED ===")
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_one_epoch():
  model.train(True)
  print(f"Epoch: {epoch+1}")
  running_loss = 0.0

  for batch_index, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)

    output = model(x_batch)
    loss = loss_function(output, y_batch)
    running_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_index % 100 == 99:
      avg_loss_across_batches = running_loss/100
      print("Batch {0}, Loss {1:.5f}".format(
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

  print("Val Loss: {0:.3f}".format(avg_loss_across_batches))
  print('***************************************************')

print(f"Starting time: {datetime.now()}")
for epoch in range(num_epochs):
  train_one_epoch()
  validate_one_epoch()

  print("=== Saving model ===")
  print(f"Epoch: {epoch}")
  torch.save(model, "../../data/prepared/model/drainLSTM_300epc.pth")

#model = torch.load("currently_trained.pth")

# to_get = 150

# with torch.no_grad():
#     predicted = model(X_test[:to_get]).to(device).to('cpu').numpy()

# plt.plot(y_test[:to_get], label='Real Log Template')
# plt.plot(predicted, label='Predicted Log Template')
# plt.xlabel('Time')
# plt.ylabel('Log Template')
# plt.legend()
# plt.show()

