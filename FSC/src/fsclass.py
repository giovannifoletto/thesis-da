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
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # setup model
    model = MatchingNetworkBERT(OUTPUT_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    support_texts = []

    print("Transforming query_set inside something usable.")
    query_text = dc(list(set(texts)))
    for text in tqdm(query_text):
        t_query_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        support_texts.append(t_query_text)

    print("Tranforming each line of the file in something compatible with our model.")
    with open(CROSS_EVAL_DATASET) as ofile:
        lines = ofile.readlines()

        for line in tqdm(lines):

            support_inputs = tokenizer(line, return_tensors="pt", padding=True, truncation=True)

            data_sample = [{
                "support_inputs": support_inputs,
                "query_inputs": support_texts
            }]


            output = model(support_inputs, support_texts)
            print(output)