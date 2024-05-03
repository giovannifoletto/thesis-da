#!/usr/bin/env python3

# Giovanni Foletto
# Implementing "multivariate_with_embeddings.ipynb" in Torch-Lightning

import lightning as L

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer

import pathlib

class UnificatedDataset(Dataset):
    def __init__(self, path, filename, tokenizer, embeddings_vector):
        self.path = pathlib.Path(path) 
        self.path = self.path / filename

        # check if file exists
        if not self.path.exists():
            print("Cannot load dataset, Path did not exists")
            return

        self.infile = open(path)
        self.lines = self.infile.readlines()

        self.tokenizer = tokenizer
        self.embeddings_vector = embeddings_vector

        self.cache = [None]*len(self.lines)
    
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        if self.cache[idx] is None:
            tokens = self.tokenizer(self.lines[idx])
            
            # Retrieve embeddings for tokens
            embeddings = self.embeddings_vector.get_vecs_by_token(tokens, lower_case_backup=True)
            self.cache[idx] = embeddings 
            return embeddings
        return self.cache[idx]


class lstmUnificatedLModel(L.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, l_in_feature):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
		
        self.l_in_feature = l_in_feature
		
		# torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
        self.l0 = nn.Linear(self.l_in_feature, self.input_size)
		# torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, 
		# dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)
        self.l1 = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
		
        self.l2 = nn.Linear(self.hidden_size, self.l_in_feature)

        self.loss_function = nn.MSELoss()
    
    def forward(self, x):
		# run the first layer
        x = self.l0(x)
		# run the LSTM layer (l1)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        x, _ = self.l1(x, (h0, c0))

        # Run the last layer => with or without the ReLU. 
        # Starting from "with"
        #x = self.l2(x)
        x = torch.relu(self.l2(x))
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                self.parameters, 
                lr=1e-3
            )
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_batch, y_batch = train_batch

        output = self(x_batch)
        loss = self.loss_function(output, y_batch)
        running_loss = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log("train_loss", running_loss, on_epoch=True)

        return running_loss
    
    def validation_step(self, val_batch, batch_idx):
        x_batch, y_batch = val_batch

        val_loss = 0.0
        with torch.no_grad():
            output = self(x_batch)
            loss = self.loss_function(output, y_batch)
            val_loss = loss.item()

        self.log("validate_loss", val_loss)

        return val_loss
    
    def backward(self, loss):
        loss.backward()
    
    def optimizer_step(self, optimizer):
        optimizer.step()

# Load GloVe '42B' embeddings
global_vectors = GloVe(name='42B', dim=300)
# Tokenize your text
tokenizer = get_tokenizer('basic_english')
# division btw train/test must be done in the files instead of the data
input = "../../data/prepared/"
test_file = "logfile_test.ndjson"
train_file = "logfile_train.ndjson"

unificated_val = UnificatedDataset(
    input, test_file,                   # file input and position
    tokenizer=tokenizer,                # tokenizer setting
    embeddings_vector=global_vectors    # embeddings
)
unificated_train = UnificatedDataset(
    input, train_file, 
    tokenizer=tokenizer, 
    embeddings_vector=global_vectors
)

val_loader = DataLoader(unificated_val)
train_loader = DataLoader(unificated_train)

## LIGHTNING SETTING
model = lstmUnificatedLModel()
trainer = L.Trainer()
trainer.fit(model, train_loader, val_loader)