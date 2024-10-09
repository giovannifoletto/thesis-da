import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW 
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, AutoTokenizer
from sklearn.preprocessing import LabelEncoder

import numpy as np
import polars as pl

from tqdm import tqdm
from copy import deepcopy as dc
import datetime, time
import random
from math import ceil
import os
from pprint import pp

from IPython import embed

from models import FineTuningDataset
from config import *

# 1. use the cosine sim after only the bert encoded string => find the more similar type of log possible
# use rnn after that

# 2. uses the bert/main.py model to have a more correct embedding of the text. after that uses RNN on what label
# bert think it is

# 3. binary classification with BERT - anomaly/normal - using the dataset as is (-2 is anomaly/reamain normal)

class MNDataset(Dataset):
    def __init__(self, tokenizer, dataframe):
        self.df = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        text = self.df["text"][idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            padding="max_length",
            truncation=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        targets = self.df["one_hot"][idx].to_numpy()
        return {
            "ids" : torch.tensor(ids, dtype=torch.long),
            "mask" : torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long)
        }

class MatchingNetwork(nn.Module):
    def __init__(self):
        super(MatchingNetwork, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-uncased')


    def forward(self):
        out = self.tokenizer

