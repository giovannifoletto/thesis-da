import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, TFDistilBertModel, DistilBertConfig, BertForSequenceClassification

from config import *
 
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
    def __init__(self, bert_model_name=BERT_MODEL_NAME, online=True):
        super(MatchingNetworkBERT, self).__init__()
        if online:
            self.bert = TFDistilBertModel.from_pretrained(bert_model_name)
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
