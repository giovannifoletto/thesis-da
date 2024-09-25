import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW 
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup, DistilBertTokenizerFast, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

from tqdm import tqdm
from copy import deepcopy as dc
import datetime, time
import random
from math import ceil
import os
from pprint import pp

from IPython import embed

from models import FineTuningDataset
from config import *

# Authors recommends
# Batch size: 16, 32
# Learning rate (Adam): 5e-5, 3e-5, 2e-5
# Number of epochs: 2, 3, 4

def finetune(labels, texts):
    # labels transformation
    # Preprocessing on labels => normalize to finetune
    #labels = minmax_scale(labels)

    labels_numpy = np.array(list(set(labels))) # get only unique labels
    n_labels = len(labels_numpy)

    # Create a LabelEncoder to map the original Label to a int64 scalar value
    le = LabelEncoder()
    le.fit(labels_numpy)
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Mapping classses/labels")
    pp(le_name_mapping)
    print("====================")

    classes = le.transform(le.classes_)
    labels_t = torch.tensor(classes)
    # labels_one_hot_encoded
    labels_one_hot_encoded = F.one_hot(labels_t[None, :], num_classes=n_labels)  
    # the labels[None, :] reshape the original numpy array
    
    embed()

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    max_length = MAX_TOKEN_LEN

    # get number of all lables to finetune on multiclass
    use_n_labels = len(set(labels))
    print(f"Labels to use: {use_n_labels}")

    # Divide labels in training and evaluations ds/dl
    training_eval_rate = 0.7
    labels_n = ceil(len(labels) * training_eval_rate)
    texts_n = ceil(len(texts) * training_eval_rate)

    print(f"lables_n: {labels_n}, texts_n: {texts_n}")

    training_lables = labels[:labels_n]
    eval_labels = labels[(labels_n+1):]

    training_texts = texts[:texts_n]
    eval_texts = texts[(texts_n+1):len(texts)]

    # Torch datasets creations
    train_ds = FineTuningDataset(
        training_texts, 
        training_lables,
        le_name_mapping,
        labels_one_hot_encoded, 
        tokenizer, 
        max_length
    )
    train_dl = DataLoader(
        train_ds, 
        batch_size=32,  # This is the recommended batch size for this model
        shuffle=True
    )

    eval_ds = FineTuningDataset(
        eval_texts, 
        eval_labels,
        le_name_mapping,
        labels_one_hot_encoded,
        tokenizer, 
        max_length
    )
    eval_dl = DataLoader(
        eval_ds, 
        batch_size=32,  # This is the recommended batch size for this model
        shuffle=True
    )

    # Initialize model, optimizer, and loss function
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = use_n_labels, # The number of output labels (multiclass classification) 
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        problem_type = "multi_label_classification"
    )

    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    print('The BERT model has {:} different named parameters.\n'.format(len(params)))
    print('==== Embedding Layer ====\n')
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== First Transformer ====\n')
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== Output Layer ====\n')
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    # This is the pytorch one, not the one from hugginface library
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    epochs = NUM_EPOCH_TRAIN

    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dl) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0, # Default value in run_glue.py
        num_training_steps = total_steps
    )

    # Train loopmatching_networks.py
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # this is done to make the training reproducible
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        embed()

        # For each batch of training data...
        for step, batch in tqdm(enumerate(train_dl)):

            # # Progress update every 10 batches.
            # if step % 40 == 0 and not step == 0:
            #     # Calculate elapsed time in minutes.
            #     elapsed = format_time(time.time() - t0)

            #     # Report progress.
            #     print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dl), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            # 'input_ids': encoding['input_ids'].flatten(),
            # 'attention_mask': encoding['attention_mask'].flatten(),
            # 'labels': label
            b_input_ids = batch["input_ids"].to(device)
            b_input_mask = batch["attention_mask"].to(device)
            b_labels = batch["labels"].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            output = model(
                b_input_ids, 
                token_type_ids=None, 
                attention_mask=b_input_mask, 
                labels=b_labels
            )

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end.
            total_train_loss += output.loss

            # Perform a backward pass to calculate the gradients.
            output.loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dl)            

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))


        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in eval_dl:

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch["input_ids"].to(device)
            b_input_mask = batch["attention_mask"].to(device)
            b_labels = batch["labels"].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                output = model(
                    b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask,
                    labels=b_labels
                )

            # Accumulate the validation loss.
            total_eval_loss += output.loss

            # Move logits and labels to CPU
            logits = output.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)


        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(eval_dl)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(eval_dl)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    print("Saving model results: ")
    output_dir = OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Saving model to {output_dir}")
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))