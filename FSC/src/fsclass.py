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

            encoding = tokenizer.encode_plus(
                line,
                add_special_tokens=True,
                max_length=max_length,

                return_token_type_ids=False,
                padding='max_length',
                truncation=True,

                return_attention_mask=True,
                return_tensors='pt', # return pytorch tensors
            )

            input_ids = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': "[MASK]"
            }


            output = model(input_ids)
            print(output)
