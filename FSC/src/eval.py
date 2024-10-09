import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

import numpy as np
import pandas as pd 

from transformers import BertModel, BertTokenizer
from IPython import embed

from itertools import cycle
from pprint import pprint as pp
from tqdm import tqdm
from models import *

MODEL_DIR = "../../data/models/bert_15ep_5e5_280ml"
MODEL_NAME = "../../data/models/bert_15ep_5e5_280ml/pytorch_model.bin"
EVAL_DS_FILENAME = "../../data/raw/flaws_cloudtrail02.ndjson"
MAX_LEN = 200
N_LABELS = 114
THRESHOLD = 0.3
BATCH_SIZE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR) 

print("Loading Model...")

model = BERTClass(n_labels=N_LABELS)

model.load_state_dict(
    torch.load(
            MODEL_NAME, 
            map_location=device,
            weights_only=True
    )
)

print("Initilaizing evaluation...")

iter_ds = StreamingDataset(EVAL_DS_FILENAME, tokenizer, MAX_LEN)
iter_dl = DataLoader(iter_ds, batch_size=BATCH_SIZE)

# targets = self.df["one_hot"][idx].to_numpy()

def transform(input):
    out_sig_f = F.sigmoid(output).numpy()
    out_sof_f = F.softmax(output, dim=1).numpy()

    out_sig = { 'max': out_sig_f.max(), 'argmax': out_sig_f.argmax()}
    out_sof = { 'max': out_sof_f.max(), 'argmax': out_sof_f.argmax()}
    
    return {
        'sigmoid': out_sig,
        'softmax': out_sof
    } 

with torch.no_grad():
    results = []
    for batch in tqdm(iter_dl): # iterate over DataLoader
        ids = batch["ids"].to(device, dtype=torch.long)
        mask = batch["mask"].to(device, dtype=torch.long)
        token_type_ids = batch["token_type_ids"].to(device, dtype=torch.long)
        output = model(ids, mask, token_type_ids)

        for i in range(BATCH_SIZE):
            res = { "text": batch["text"][i] }
            res.update(transform(output[i]))
            #pp(res)
            results.append(res)
        if len(results) % 1000 == 0:
            df = pd.DataFrame(results)
            df.to_parquet(f"../results/evaluation_model_matteo_random_{len(results)}.parquet")
            print(f"Saving {len(results)} results.")
    embed()

