import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertConfig, BertForSequenceClassification

from config import *

from IPython import embed
 
# Configure DistilBERT's initialization
BertConfig = DistilBertConfig(
    dropout=DISTILBERT_DROPOUT, 
    attention_dropout=DISTILBERT_ATT_DROPOUT, 
    output_hidden_states=True
)

# would be better
# # Combine the training inputs into a TensorDataset.
# dataset = TensorDataset(input_ids, attention_masks, labels)
class FineTuningDataset(Dataset):
    def __init__(self, texts, labels, l_mapping, l_one_hot_encoded, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.l_mapping = l_mapping
        self.l_one_hot_encoded = l_one_hot_encoded
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # get label corresponding to the initial ds
        orig_label = self.labels[idx]
        # get mapped label
        mapp_label = self.l_mapping[orig_label].item()
        label = self.l_one_hot_encoded[mapp_label]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,

            return_token_type_ids=False,
            padding='max_length',
            truncation=True,

            return_attention_mask=True,
            return_tensors='pt', # return pytorch tensors
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

class BERTFineTuner(nn.Module):
    def __init__(self, bert_model_name=BERT_MODEL_NAME, num_labels=2):
        super(BERTFineTuner, self).__init__()

        self.num_labels = num_labels
        self.bert = BertForSequenceClassification.from_pretrained(
            bert_model_name,
            num_labels=num_labels,
            output_attentions = False,
            output_hidden_states = False,
        )
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output        
        x = self.dropout(pooled_output)
        logits = self.fc(x)

        return logits
    

class MatchingNetworkBERT(nn.Module):
    def __init__(self, bert_model_name=BERT_MODEL_NAME):
        super(MatchingNetworkBERT, self).__init__()
        
        self.bert = BertForSequenceClassification.from_pretrained(
            bert_model_name,
            output_hidden_states=True
        )

        # You can add more layers here if needed
        self.fc = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def encode(self, tokenized_input):
        return self.bert(**tokenized_input).hidden_states[-1][:, 0, :]

    def forward(self, support_set, query_set):
        # Embed support set using BERT
        support_embeddings = F.relu(self.fc(support_set))

        matmul = F.relu(torch.matmul(support_embeddings, query_set.transpose(0, 1)))
        probabilities = F.relu(self.softmax(matmul))

        return probabilities

