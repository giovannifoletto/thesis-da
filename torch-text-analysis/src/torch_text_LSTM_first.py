# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# Giovanni Foletto
# > Complete pipeline to understand, tokenize and try to classificate different logs.
# > After that, this scrips try to arrange data in a way that is useful for the LSTM model to work
# > and start to train the model. 

# > This implementation does not use the entire logs, and are other methods that could works better
# > and faster in this script (like the whole working on data, that could be done better and faster)
# > using the unificated.ndjson

# > The script is divided in two parts,
# > 1. The first here is when the data are get in and then evaluate/manipulated and so they are prepared for the ML algorithm
# > 2. the second part of the script is in the second file, in this directory, the one that elaborate the data and try to run the 
# >     model.

import numpy as np
import pandas as pd
import polars as pl

import torch
import torch.nn as nn
import torchtext as tt
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import seaborn as sns

import json

from collections import Counter
from copy import deepcopy as dc

SELECTED = ["eventTime", "userAgent", "errorMessage", "eventType", "errorCode", "sourceIPAddress", "eventName", "eventSource", "awsRegion"]

df = pl.read_ndjson("../../data/raw/unificated.ndjson")
df = df.select(SELECTED)
df = df.fill_null(strategy="forward")

df.head(2)

# DO NOT RUN IT NOW, the initial parsing breaks if the datatime is converted now
# Is more valueable to be run later
#df = df.with_columns(pl.col("eventTime").str.to_datetime())
#df.head(2)

# Example of what we want at scale:
#toker = get_tokenizer("basic_english")
#toker(
#    "".join(
#        [str(x) for x in df.select(pl.col("*").exclude("sourceIPAddress", "eventTime"))[0].to_numpy()]
#        ).replace("[", "").translate({
#            ord("["): None,
#            ord("]"): None,
#            ord("."): None,
#            ord(","): None,
#            ord("'"): None,
#        })
#    )
#

#df.to_numpy()[0, 1:-1]

toker = get_tokenizer("basic_english")

TRANSLATION_VOCABULARY = {
              ord("["): None,
              ord("]"): None,
              ord("."): None,
              ord(","): None,
              ord("'"): None,
            }

def create_vocabulary(dataframe):
  counter_obj = Counter()
  dataset = dataframe.clone().select(pl.col("*").exclude("sourceIPAddress", "eventTime"))
  for d in dataset.rows():
    try:
      txt = " ".join([x for x in d]).translate(TRANSLATION_VOCABULARY)
    except:
      continue
    split_and_lowered = toker(txt)
    counter_obj.update(split_and_lowered)

  result = tt.vocab.vocab(counter_obj, min_freq=1)
  return result

vocab = create_vocabulary(df)
vocab.set_default_index(-1)

print(len(vocab))

torch.save(vocab, "vocab_vers_2.pth")

# To recovered from previously saved vocabulary
#vocab = torch.load("vocab_vers_2.pth")

# Only for testing
# for i in range(len(vocab)):
#   print(vocab["linux/440-157-generic"])
#   break

def from_tuple_to_token(tup):
  #print(tup[-1])
  try:
    txt = " ".join([x for x in tup]).translate(TRANSLATION_VOCABULARY)
  except Exception as e:
    txt = "No data"
    print(f"WARNING: skipping {e} - {tup}")
  res = []
  norm = toker(txt)
  for i in norm:
    res.append(str(vocab[i]))
  return res

# select(
#     pl.col("*").exclude("sourceIPAddress", "eventTime")
#     )

# This section create a new column tha contains the entire information present in the first place in the whole dataset
# and then return it only with "eventTime"/"sourceIPAddress", "concat_string"
df = df.with_columns(
        pl.concat_str(
            pl.col("*").exclude("sourceIPAddress", "eventTime"),
            separator=" "
          ).alias("concatenated_list") # col("*").exclude("sourceIPAddress", "eventTime").arr.
    )
df.head(2)

sample = df.select(pl.col("eventTime", "sourceIPAddress", "concatenated_list"))
sample.head(2)

del df

# This section takes in the new dataframe (sample) and run on every line
# 1. The tokenizer
# 2. The translated and pre-processor
# 3. collect the results in the same dataset, and save it

# f = open("output_tokenized_data.json", "w")
new_data = []
for i in sample.rows(named=True):
  tokenized_data = dict(i)
  tokens = from_tuple_to_token(i["concatenated_list"])
  tokenized_data["tokenized_value"] = torch.tensor([int(x) for x in tokens if x != -1]).float()
  #print(len(tokenized_data["tokenized_value"]))
  t_shape = tokenized_data["tokenized_value"].shape
  tokenized_data["tokenized_value"] = tokenized_data["tokenized_value"] @ torch.ones(t_shape).float()
  tokenized_data["tokenized_value"] = tokenized_data["tokenized_value"].item()

  new_data.append(tokenized_data)
  #print(tokenized_data["tokenized_value"].item())
  # f.write(json.dumps(tokenized_data))
  # f.write("\n")
  #break

df_new_data = pl.DataFrame(new_data)
df_new_data.head()
#f.close()

# This was useful to colab, since saving locally was faster tha saving it on drive
#df_new_data.write_csv("output_tensor_data.csv")

df_new_data.write_csv("../../data/prepared/output_tesor_data.csv")