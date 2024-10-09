# Giovanni Foletto - thesis-da/FSC/src/models.py
# this file contains all the models and Dataset classes needed to 
# finetune the model, train it, evaluate and use the given model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import DistilBertConfig, BertForSequenceClassification, BertModel

from config import *

from IPython import embed
from copy import deepcopy as dc
from itertools import cycle

# FineTuningDataset: collate togheter an ad-hoc created dataframe
# able to elaborate inputs from within itself.
class FineTuningDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        super(FineTuningDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.df = dc(dataframe)

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

# BertModel to finetune.
class BERTClass(torch.nn.Module):
    def __init__(self, n_labels):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-uncased')
        self.l2 = nn.Dropout(0.3) # Try this .1
        # this is the layer that return the labels => must have the same number of labels the training set is emitting
        self.l3 = nn.Linear(768, n_labels)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

# create an iterable dataset
# this dataset will iterate on a file that contains each line a 
# element that should be processed
class StreamingDataset(IterableDataset):
    def __init__(self, file, tokenizer, max_len):
        super(StreamingDataset).__init__()
        self.file = file
        self.tokenizer = tokenizer
        self.max_len = max_len

    def parse_file(self, filename):
        with open(filename) as file:
            for line in file:

                inputs = self.tokenizer.encode_plus(
                    line,
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
                text = line
                
                yield {
                    "ids": torch.tensor(ids),
                    "mask": torch.tensor(mask),
                    "token_type_ids": torch.tensor(token_type_ids),
                    "text": text
                }

    def get_stream(self, filename):
        return cycle(self.parse_file(filename))

    def __iter__(self):
        return self.get_stream(self.file)
    
