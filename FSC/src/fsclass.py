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

# ########################
# Start Few-shot classification using Matching Networks
# ########################

# no need to load the same thing again
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# input the evaluation dataset and ask the model to associate to a finetuned label
# (this is not entirely correct, but it can fair enought for now)

def fsclass(labels, texts):
    # setup
    tokenizer = DistilBertTokenizerFast.from_pretrained('bert-base-uncased')
    max_length = MAX_TOKEN_LEN

    # setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    support_texts = []
    with open(EVAL_DATASET) as ev_file:
        support_text = ev_file.readlines()

    # Setting Query text with only the available labels
    query_text = dc(set(labels))

    # Tokenize the input texts
    support_inputs = tokenizer(support_texts, return_tensors="pt", padding=True, truncation=True)
    query_inputs = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True)

    # Create the model
    model = MatchingNetworkBERT(
        bert_model_name="../results/model_299.pth",  # This should be changed with the "evaluated" best model not only with the last
        online=False
        )

    # Get the output
    output = model(support_inputs, query_inputs)
    print(output)