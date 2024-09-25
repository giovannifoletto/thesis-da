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
    model = BertForSequenceClassification.from_pretrained(OUTPUT_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("Starting Matching networks.")

    outputs = []
    with open(EVAL_DATASET) as ofile:
        lines = ofile.readlines()

        for line in tqdm(lines):

            support_inputs = tokenizer(line, return_tensors="pt", padding=True, truncation=True)
            support_inputs.to(device)

            with torch.no_grad():
                logits = model(**support_inputs).logits
            
            predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]
            num_labels = len(model.config.id2label)

            counter = 1
            for pci in predicted_class_ids:
                print(f"Prediction {counter}: {pci} over #{num_labels} num of labels")
            #print(model.config.id2label[predicted_class_id])      


            # embed()
            #     support_inputs = model.encode(support_inputs)
            #     support_inputs.to(device)
            #     output = model(support_inputs, support_texts)
            # outputs.append(output)

            # print(output.max(), output.argmax())
            # output_label = labels[output.argmax()]
            # print(f"This model returned label: {output_label} with probability: {output.max()}")
