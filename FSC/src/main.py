# Giovanni Foletto
# Following => https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import numpy as np
import polars as pl
import time, datetime

import os
from copy import deepcopy as dc
import json
from tqdm import tqdm
import random
from pprint import pp
from IPython import embed

from config import *
from model import *

# Preparing for TPU usage
# import torch_xla
# import torch_xla.core.xla_model as xm
# device = xm.xla_device()
# device = ''
# if torch.cuda.is_available():
#   device = torch.device('cuda')
# elif torch.xla.core.is_available(): # this methods would probably do not exists
#   device = xm.xla_device()
# else:
#   device = torch.device('cpu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using {device} as training device')

# Config
# Sections of config

MAX_LEN = 280
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 1e-5
SEED_VALUE = 42

#####################
# DATA PREPARATION
#####################

print("Importing datasets...")
# dataset original
do = []
# import dataset here
with open(DATASET_WITH_LABEL) as text_file:
    text_lines = text_file.readlines()
    for line in tqdm(text_lines):
        jo = json.loads(line)
        do.append({"text": line, "columns": jo})

df = pl.DataFrame(do, strict=False)
df = df.with_columns(df.select("columns").unnest("columns")["label"]).drop("columns")

# Create labelMapping
# unique labels
u_labels = df.select("label").unique() # get only unique labels
n_labels = u_labels.shape[0]

print(f"Using {n_labels} as number of labels in this dataset")

# Create a LabelEncoder to map the original Label to a int64 scalar value
l_numpy = u_labels.to_numpy().reshape((-1, 114)) # make the array linear
l_numpy = np.squeeze(l_numpy, axis=0) # make monidimensional

print("Mapping classses/labels...")
le = LabelEncoder()
le.fit(l_numpy)
classes = le.transform(le.classes_)
# create a set/mapping from original label=>0-114 labels
# le_labels_mapping = list(zip(le.classes_, classes)) 
le_labels_mapping = pl.DataFrame(
                        list(zip(le.classes_, classes))
                    ).transpose().rename({
                        "column_0": "label",
                        "column_1": "m_label"
                    })

print("Creating one-hot encoding for each label...")
labels_t = torch.tensor(classes, dtype=torch.long)
labels_one_hot = F.one_hot(labels_t[None, :], num_classes=n_labels)
labels_one_hot = labels_one_hot.squeeze()
le_labels_mapping = le_labels_mapping.with_columns(
                        pl.Series(
                            labels_one_hot.squeeze().numpy()
                        ).alias("one_hot")
                    )

print("Joining labels/labels_text...")
df = df.join(le_labels_mapping, on="label").rename(
    {
        "label": "label_name",
        "m_label": "label"
    }
)

######################
# TRAINING PREPARATION
######################

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Creating the dataset and dataloader for the neural network

train_size = TRAIN_BATCH_SIZE

train_dataset = df.sample(fraction=train_size, seed=SEED_VALUE).with_row_index()
test_dataset = df.with_row_index().join(train_dataset, on=["index"], how="anti")

train_dataset.drop_in_place("index")
test_dataset.drop_in_place("index")

print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

train_set = FineTuningDataset(train_dataset, tokenizer, MAX_LEN)
test_set = FineTuningDataset(test_dataset, tokenizer, MAX_LEN)

train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=VALID_BATCH_SIZE, shuffle=True, num_workers=0)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train(epoch):
    print("    --- Training ---")
    model.train()
    total_train_loss = 0

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    t0 = time.time()
    for data in tqdm(train_loader):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        # Reset previusly calculated gradient. Pytorch doesn't do it themselve
        model.zero_grad()
        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = nn.BCEWithLogitsLoss()(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
        optimizer.zero_grad()
        optimizer.step()
        total_train_loss += loss.item()
    
    if device != "cpu":
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

    avg_train_loss = total_train_loss / len(train_loader)
    training_time = format_time(time.time() - t0)
    
    print(f'Epoch: {epoch+1}, Total Loss: {total_train_loss/len(train_loader)}')

def validation(epoch):
    print("    --- Validation ---")

    model.eval()
    fin_targets=[]
    fin_outputs=[]

    with torch.no_grad():
        for data in tqdm(test_loader):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    print(f'Validation: {epoch}')
    return fin_outputs, fin_targets

print("Instantiating model and start trainig...")
model = BERTClass(n_labels=n_labels)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr = LEARNING_RATE)

m_accuracy = 0

# Make training repruducible
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)
torch.cuda.manual_seed_all(SEED_VALUE)

for epoch in range(EPOCHS):

    print(f'******** Started Finetuning {epoch+1} ********')
    train(epoch)
    outputs, targets = validation(epoch)

    outputs = np.array(outputs) >= 0.5 
    accuracy = metrics.accuracy_score(targets, outputs)
    # probably we have to set `zero_division`
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f"    Accuracy Score = {accuracy}")
    print(f"    F1 Score (Micro) = {f1_score_micro}")
    print(f"    F1 Score (Macro) = {f1_score_macro}")

    if accuracy > m_accuracy:
        print('This model has higher accuracy. Saving...')
    
        output_dir = OUTPUT_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(f"Saving model to {output_dir}")
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(output_dir+'/tokenizer/')

        config_to_save = {
            "f1-score_macro": f1_score_macro,
            "f1-score_micro": f1_score_micro,
            "accuracy": accuracy,
            "n_labels": n_labels,
            "traning_info": {
                "epoch_used": epoch+1,
                "max_token_len": MAX_LEN,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "valid_batch_size": VALID_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "seed_value": SEED_VALUE
            }
        }

# retry with
# different parameters in the loading, training, batching, epochs
# changing the np.array(outputs) >= 0.6
# https://www.kaggle.com/code/neerajmohan/fine-tuning-bert-for-text-classification
# https://medium.com/codex/fine-tune-bert-for-text-classification-cef7a1d6cdf1
# https://ellielfrank.medium.com/understanding-the-f1-score-55371416fbe1
# https://encord.com/blog/f1-score-in-machine-learning/

# 9 epochs of validation
# F1-score: 0,89 (going over change only some millimesimal, not much)
# this is probably the limit.
# We can get the accuracy up a little (81%)
