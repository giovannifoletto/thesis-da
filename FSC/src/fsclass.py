import torch
from transformers import AutoTokenizer, BertForSequenceClassification

from sklearn.preprocessing import minmax_scale
import numpy as np
import polars as pl

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

    # this is wrong
    print("Import vocabulary")
    support_texts = []
    query_labels = dc(list(set(labels)))
    for ql in tqdm(query_labels):
        idx = np.where(labels == ql)[0]
        if len(idx) > 0:
            idx = idx[0].item()
        else:
            idx.item()
        t_query_text = tokenizer(texts[idx], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            t_query_text.to(device)
            output = model.encode(t_query_text)

        support_texts.append(output[0].numpy())

    # This is very wrong => we have to do:
    # - this array must go, a matrix has to come
    # - the matrix should have on each colums one of the embeddings => 114x768    

    print(f"Vocabulary len: {len(support_texts)}")

    # /home/giovannifoletto/Documents/programmazione/thesis-da/FSC/src/fsclass.py:54: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at ../torch/csrc/utils/tensor_new.cpp:278.)
    # support_texts = torch.tensor(support_texts).to(device)
    support_texts = torch.tensor(support_texts)
    support_texts.to(device)

    #embed()

    print("Starting Matching networks.")

    outputs = []
    with open(CROSS_EVAL_DATASET) as ofile:
        lines = ofile.readlines()

        for line in tqdm(lines):

            support_inputs = tokenizer(line, return_tensors="pt", padding=True, truncation=True)
            support_inputs.to(device)

            with torch.no_grad():
                support_inputs = model.encode(support_inputs)
                support_inputs.to(device)
                output = model(support_inputs, support_texts)
            outputs.append(output)

            print(output.max(), output.argmax())
            output_label = labels[output.argmax()]
            print(f"This model returned label: {output_label} with probability: {output.max()}")
