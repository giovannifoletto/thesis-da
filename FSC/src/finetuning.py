import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, DistilBertTokenizerFast

from sklearn.preprocessing import minmax_scale
import numpy as np

import json
from tqdm import tqdm
from copy import deepcopy as dc
import datetime

from IPython import embed

from models import FineTuningDataset, BERTFineTuner, MatchingNetworkBERT
from config import *

# Authors recommends
# Batch size: 16, 32
# Learning rate (Adam): 5e-5, 3e-5, 2e-5
# Number of epochs: 2, 3, 4

def finetune(labels, texts):
    tokenizer = DistilBertTokenizerFast.from_pretrained('bert-base-uncased')
    max_length = MAX_TOKEN_LEN

    # Torch datasets creations
    dataset = FineTuningDataset(texts, labels, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = BERTFineTuner(
        num_labels=len(set(labels))
    )
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Train loopmatching_networks.py
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    epochs = NUM_EPOCH_TRAIN
    for epoch in tqdm(range(epochs)):
        print(f"Traning epoch: {epoch}")
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
        # Saving intermediate step
        torch.save(model.state_dict(), f'../results/model_weights_{epoch}.pth')
        torch.save(model, f'../results/model_{epoch}.pth')
        with open(f"../results/log_epoch_{epoch}.log", "w") as logfile:
            # log object
            lo = {
                "datetime": datetime.now(),
                "loss": loss, 
                "outputs": outputs,
            }
            json.dump(logfile, lo)