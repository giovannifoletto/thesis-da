import torch
from transformers import AutoTokenizer, BertForSequenceClassification

from sklearn.preprocessing import minmax_scale
import numpy as np

from tqdm import tqdm
from copy import deepcopy as dc
import datetime

from IPython import embed

from models import BERTFineTuner, MatchingNetworkBERT
from config import *

# ########################
# Start Few-shot classification using Matching Networks
# ########################

# no need to load the same thing again
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def fsclass(labels, texts):
    # setup
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    max_length = MAX_TOKEN_LEN

    model = BertForSequenceClassification.from_pretrained(OUTPUT_DIR)

    # setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with open(CROSS_EVAL_DATASET) as ofile:
        lines = ofile.readlines()

        for line in lines:
            output = model(line)
            print(output)
