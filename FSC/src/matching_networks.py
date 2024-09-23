from sklearn.preprocessing import minmax_scale
import numpy as np

import json
from tqdm import tqdm
from copy import deepcopy as dc

from config import *

from finetuning import finetune
from fsclass import fsclass

# Giovanni Foletto
# First phase: finetuning with the labeled dataset
# Second phase: few-shot classification
    
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

# to check if there are enought labels with text (they have to
# be the exact number)
assert(len(texts) == len(labels))

# Preprocessing on labels => normalize to finetune
labels = minmax_scale(labels)

finetune(labels, texts)
fsclass(labels, texts)


# This output will be the statistical representation of what the model thinks is the better label
# to assign the log.

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
