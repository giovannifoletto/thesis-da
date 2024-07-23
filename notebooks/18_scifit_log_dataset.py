#!/usr/bin/env python
# coding: utf-8

# # Try SciFit with Log dataset
# 
# The goal is to understand if the SciFit model can works with different and more complex dataset, like the logs we are using.
# The approach is similar to the LogPrecis application, but more hand-down approach with low/none-application specific analysis
# blind few-shot classification operations.

# @author: Giovanni Foletto
# @date: July 22, 2024

import polars as pl
import numpy as np
import pandas as pd

import json
from copy import deepcopy as dp

from datasets import load_dataset, IterableDataset 
from datasets import Dataset as hfDataset
from sentence_transformers.losses import CosineSimilarityLoss

import torch
import torch.nn as nn
import torchtext as tt
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader

from setfit import SetFitModel, Trainer, TrainingArguments

import math

log_location = {
    "train": "/home/rising/2024-06-21-category-1-sorted-cplabels.json",
    "evaluation": "/home/rising/2024-06-21-random-luis-matteo.json"
}

def import_from_json(log_location):
	data = list()
	with open(log_location) as log_file:
		log_lines = log_file.readlines()
		for line in log_lines:
			old_obj = json.loads(line)

			new_obj = dict()

			try:
				new_obj['label'] = old_obj['label']
				new_obj['log'] = dp(old_obj)
				new_obj['log'].pop('label', None)

				new_obj['log'] = json.dumps(new_obj['log'])
				new_obj['text-label'] = 'label-n-' + str(old_obj['label'])
			except KeyError:
				new_obj['log'] = dp(old_obj)
				new_obj['log'].pop('label', None)

				new_obj['log'] = json.dumps(new_obj['log'])

			data.append(new_obj)

	return data

df = pl.DataFrame(import_from_json(log_location["train"]))

print("==== DATASET HEAD ====")
print(df.head())

print("==== COUNT LABELS ====")
print(df.select(pl.col("label").value_counts()).unnest("label"))

train_over_test_ratio = math.floor(0.75 * df.shape[0])

print("RATIO TEST/TRAIN: ", train_over_test_ratio)

df_train = dp(df.head(train_over_test_ratio))
df_test = dp(df.head(-train_over_test_ratio))

del df

print("==== SHAPES ====")
print("df_train.shape", df_train.shape)
print("df_train.shape", df_test.shape)

df_train_hf = hfDataset.from_dict(df_train.to_dict())
df_test_hf = hfDataset.from_dict(df_test.to_dict())

print("DS_Train:\n", df_train_hf)
print("DS_Test:\n", df_test_hf)

model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

args = TrainingArguments(
    batch_size=16,
    num_epochs=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=df_train_hf,
    eval_dataset=df_test_hf,
    metric="accuracy",
    column_mapping={"log": "text", "label": "label"}  # Map dataset columns to text/label expected by trainer
)

trainer.train()

metrics = trainer.evaluate()
print("==== METRICS ====")
print(metrics)