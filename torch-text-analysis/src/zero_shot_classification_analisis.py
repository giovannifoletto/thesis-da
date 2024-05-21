from transformers import pipeline
from tqdm import tqdm
import json

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

log_description = ["malicious", "benign"]
openfile = open("../../data/raw/unificated.ndjson")
lines = openfile.readlines()[83:15000]

#outputfile = open("./results_transformers_line_by_line.res", "w")
outputfile = open("./results_transformers2.res", "a")

log_description = ["malicious", "benign"]
log_description2 = ["anomalous", "normal"]
for line in tqdm(lines):
    #results = classifier(line, log_description)
    results2 = classifier(line, log_description2)

    # for label, score in zip(results["labels"], results["scores"]):
    #     outputfile.write("\n")
    # print(results)
    outputfile.write(json.dumps(results2))
    outputfile.write("\n")

openfile.close()
outputfile.close()
#outputfile2.close()