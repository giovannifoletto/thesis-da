from transformers import pipeline
from tqdm import tqdm
import json

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Result1: using ["malicious", "benign", "undecided"]
# Result3: using ["malicious", "benign", "undecide"] => different dataset (matteo/luis)

# LLM are not ok for this:
# 1. time to execute analysis
# 2. no correct solution about anomalies, but only semantically correlation.

log_description = ["malicious", "benign", "undecided"]
openfile = open("/home/rising/2024-06-21-random-luis-matteo.json")
lines = openfile.readlines()

#outputfile = open("./results_transformers_line_by_line.res", "w")
outputfile = open("../../data/prepared/zero_shot_classification/result_3_random_luis_matteo.txt", "a")
for line in tqdm(lines):
    results = classifier(line, log_description)

    outputfile.write(json.dumps(results))
    outputfile.write("\n")

openfile.close()
outputfile.close()
#outputfile2.close()