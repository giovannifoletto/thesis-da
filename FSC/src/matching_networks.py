import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW

from sklearn import preprocessing
import numpy as np

import json
from tqdm import tqdm
from copy import deepcopy as dc
import datetime

from IPython import embed

# Giovanni Foletto
# First phase: finetuning with the labelled dataset
# Second phase: few-shot classification

DATASET_WITH_LABEL = "/home/rising/2024-06-21-category-1-sorted-cplabels.json"
MAX_TOKEN_LEN = 128
NUM_EPOCH_TRAIN = 5
EVAL_DATASET = "/home/rising/2024-06-21-random-luis-matteo.json"

class FineTuningDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        print(self.labels[idx])
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

class BERTFineTuner(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=2):
        super(BERTFineTuner, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        #print("BERT output:", outputs)

        pooled_output = outputs.pooler_output
        

        #print("Pooled output", pooled_output)
        
        x = self.dropout(pooled_output)
        #print("X: ", x)

        logits = self.fc(x)
        #print("Logits:", logits)
        return logits

# data elaboration to split json in text and label
print("Importing datasets")
texts = []
labels = []

# import dataset here
with open(DATASET_WITH_LABEL) as text_file:
    text_lines = text_file.readlines()
    for line in tqdm(text_lines):
        jo = json.loads(line)
        text = dc(jo)
        text.pop('label')
        texts.append(json.dumps(text))
        labels.append(jo['label'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = MAX_TOKEN_LEN

assert(len(texts) == len(labels))

# Preprocessing on labels => normalize to finetune
# labels = np.array(labels).reshape(-1, 1)
# scaler = preprocessing.MinMaxScaler().fit(labels)
# labels = scaler.transform(labels)
# labels = labels.reshape(1, -1)[0]
from sklearn.preprocessing import minmax_scale
labels = minmax_scale(labels)

#print(labels.shape)

#embed()


# Torch datasets creations
dataset = FineTuningDataset(texts, labels, tokenizer, max_length)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize model, optimizer, and loss function
model = BERTFineTuner()
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Train loopmatching_networks.py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = NUM_EPOCH_TRAIN
for epoch in tqdm(range(epochs)):
    print(f"Traning epoch: {epoch}")
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        #print("Batch: ", batch)
        #print("output: ", outputs)
        loss = criterion(outputs, labels)
        print("Loss: ", loss)

        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    # Saving intermediate step
    torch.save(model.state_dict(), f'../results/model_weights_{epoch}.pth')
    torch.save(model, f'../results/model_{epoch}.pth')
    with open(f"../results/log_epoch_{epoch}.log") as logfile:
        # log object
        lo = {
            "datetime": datetime.now(),
            "loss": loss, 
            "outputs": outputs,
        }
        json.dump(logfile, lo)

# ########################
# end of Finetuning
# ########################

# ########################
# Start Few-shot classification using Matching Networks
# ########################

class MatchingNetworkBERT(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', online=True):
        super(MatchingNetworkBERT, self).__init__()
        if online:
            self.bert = BertModel.from_pretrained(bert_model_name)
        else:
            self.bert = torch.load(bert_model_name, weights_only=False)
        # You can add more layers here if needed
        self.fc = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size) 

    def forward(self, support_set, query_set):
        # Embed support set using BERT
        support_embeddings = self.bert(**support_set).last_hidden_state[:, 0, :]  # Take the [CLS] token embedding
        support_embeddings = F.relu(self.fc(support_embeddings)) 

        # Embed query set using BERT
        query_embeddings = self.bert(**query_set).last_hidden_state[:, 0, :]  # Take the [CLS] token embedding
        query_embeddings = F.relu(self.fc(query_embeddings)) 

        # Calculate similarity between query and support embeddings
        similarity = torch.matmul(query_embeddings, support_embeddings.transpose(0, 1))

        # Softmax to get probabilities
        probabilities = F.softmax(similarity, dim=1)

        return probabilities

# no need to load the same thing again
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# input the evaluation dataset and ask the model to associate to a finetuned label
# (this is not entirely correct, but it can fair enought for now)
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

# This output will be the statistical representation of what the model thinks is the better label
# to assign the log.

# - Target 5 is out of bound (too big)

# Evaluation is done wrong:

# 1. divide the labels dataset in 2 parts (70/30)
# 2. use the first 70% to train and finetune (check if this is randomized)
# 3. use the second part for evaluation (this will be not as good since it is evaluated on already trained data)
# 4. import the new dataset and evaluate with that 
#       PROBLEM: with this new DS there is a method in which I can understand if it correct or not? How?

# Other problems:
# - no recover possibilities (if the script die, we have to restart. BUT: the model is saved each iteration?)
#       PROBLEM: is saved correctly? Other information that could be useful in the log
# - the log in this phase are directly truncate (without even adding the closing })
#       PROBLEM: resolve that, is not coherent with the presented solution possibilities
# - we need more flexibility to run this model => create a CLI would be better.
