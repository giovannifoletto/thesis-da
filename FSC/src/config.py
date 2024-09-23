BERT_MODEL_NAME = 'bert-base-uncased'
DATASET_WITH_LABEL = "../../data/raw/2024-06-21-category-1-sorted-cplabels.json"
EVAL_DATASET = "../../data/raw/2024-06-21-random-luis-matteo.json"
MAX_TOKEN_LEN = 128
NUM_EPOCH_TRAIN = 4 # this is the recommended for BERT
LEARNING_RATE = 5e-5
OUTPUT_DIR = "../results/"

DISTILBERT_DROPOUT = 0.2
DISTILBERT_ATT_DROPOUT = 0.2

CROSS_EVAL_DATASET = "../../data/raw/flaws_cloudtrail02.ndjson"