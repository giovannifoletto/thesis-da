#!/usr/bin/env python3
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import glob
import json
from tqdm import tqdm

from Types.Types import cloudtrails_log_schema

option = dict()
option["train_dim"] = 6e-1 
option["cross_dim"] = 3e-1
option["test_dim"]  = 3e-1
option["to_maintain"] = [
    "userAgent",
    "eventType",
    "eventName",
    "eventSource",
    "recipientAccountId",
    "awsRegion",
    "eventVersion", # TO INVESTIGATE WHY IS LIKE THIS
    #"errorCode",
]

class Cooking:
    def __init__(self, infile, voc_dir, tokenized_data) -> None:
        self.df = pl.read_ndjson(infile, schema=cloudtrails_log_schema)
        self.voc_dir = Path(voc_dir)
        self.tokenized_data = Path(tokenized_data)

        if self.voc_dir.is_dir():
            print(f"Outdir exists: {self.voc_dir}")
        else:
            print(f"Did not find {self.voc_dir} directory, creating one")
            self.voc_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loaded dataframe {self.df.shape}")
        print(self.df.columns)
        self.df.fill_null("None")
        self.df.fill_nan("None")
        self.df = self.df.select(option["to_maintain"])
        print(f"Filtered Dataset: {self.df.shape}")

        self.voc = dict()

        # df_dim_train = self.df.shape[0] * option["train_dim"]
        # df_dim_cross = self.df.shape[0] * option["cross_dim"] + df_dim_train
        # df_dim_test  = self.df.shape[0] * option["test_dim"] + df_dim_cross
        # self.df_train = self.df[:df_dim_train]
        # self.df_cross = self.df[df_dim_train:df_dim_cross]
        # self.df_test  = self.df[df_dim_cross:df_dim_test]

        # print(option, df_dim_train, df_dim_cross, df_dim_test) # Check if this is correct

    def createVocabularies(self):
        for key in option["to_maintain"]:
            unique_set = self.df.select(key).unique()
            dim = unique_set.shape[0]
            print(f"Found unique in {key}: {unique_set.shape}")

            unique_set = unique_set.fill_null("None")
            unique_set = unique_set.drop_nulls()
            print(f"Null count: {unique_set.null_count().item()}, {unique_set.select(pl.col('*')).filter(pl.col(key).is_null())}")

            np_array = np.array([i for i in range(dim)])
            new_df = pl.concat(
                [unique_set, pl.DataFrame(np_array, schema={
                        f"{key}-token": pl.Int64
                    })], 
                how="horizontal")

            print(f"Accosting with index-made token for {key}: {new_df.shape}/{new_df.dtypes}")

            output_path = self.voc_dir / f"{key}.csv"

            new_df.write_csv(output_path, separator="|")

    def return_tokenized(self):

        print(f"Starting with dictionary: {self.df.shape}")

        for key in option["to_maintain"]:
            
            # Load Vocabulary
            filename = self.voc_dir / f"{key}.csv"
            print(f"Loading Vocabulary {key}: {filename}")
            to_join = pl.read_csv(filename, separator="|", schema={
                f"{key}": pl.Utf8,
                f"{key}-token": pl.Int64
            })

            # Prepare and make join
            print(f"Joining with {key}, {to_join.shape}/{to_join.columns}")
            #print(f"Null count: {to_join.null_count()}, {to_join.select(pl.col('*')).filter(pl.col(key).is_null())}")

            self.df = self.df.join( to_join, on=key )
            print(f"Saving temp token-only copy")
            temp_copy_loc = self.voc_dir / f"last_join_tmp_copy.csv" 
            self.df.write_csv(temp_copy_loc, separator="|")
            print(f"Finished join with {key}: {self.df.shape}")

        print(f"After join: {self.df.shape}")
        self.df.select(pl.col("*").exclude(option["to_maintain"])).write_csv(self.tokenized_data)
    
class Classification:
    def __init__(self, infile, outfile, seed) -> None:
        infile = open(infile)
        self.lines = infile.readlines()
        self.prolog = self.lines[0].rstrip()
        self.seed = seed

        self.outtmp = Path(outfile)
        
        if self.outtmp.exists():
            print(f"Outfile exists: {self.outtmp}")
        else:
            print(f"Did not find {self.outtmp} file, creating one")
            open(self.outtmp, "w").close()

        self.outfile = open(self.outtmp, "w")
    
    def calculate_dot_product(self):
        self.outfile.write(f"{self.prolog},dot\n")
        for line in tqdm(self.lines[1:]):
            line = line.rstrip()
            arr = np.array([int(i) for i in line.split(',')])
            dot = np.dot(self.seed, arr)
            self.outfile.write(f"{line},{dot}\n")
    
    def __del__(self):
        self.outfile.close()

def transform_data_to_ndjson(path_to_glob, path_to_res):
    outfile = open(path_to_res + "unificated.ndjson", "w")
    for f in tqdm(glob.glob(path_to_glob)):
        with open(f, "r") as openj:
            line = openj.readlines()
            assert len(line) == 1
            
            jf = json.loads(line[0])
            for je in jf["Records"]:
                outfile.write(json.dumps(je))
                outfile.write("\n")
    outfile.close()

def tokenization_error(infile):
    df = pl.read_csv(infile)
    dim = df.shape
    print(f"SHAPE: {dim}")

    correct = df.select(pl.col("dot").value_counts()).unnest("dot").filter(pl.col("count") == 1).count()["dot"][0]
    incorrect = df.select(pl.col("dot").value_counts()).unnest("dot").filter(pl.col("count") != 1).count()["dot"][0]
    print(f"Correct-ness: {correct/dim[0]}")
    print(f"INcorrect-ness: {incorrect/dim[0]}")
    

if __name__ == "__main__":
    # 1. Divide data-set in train (60%), cross-entropy(30%), test(30%) 
    # 2. Create vocabulary
    # 3. Transform words in number
    # 4. Evaluate

    path_to_glob = "../data/*/flaws*.json"
    path_to_res = "../data/raw/"

    if not Path(path_to_res + "unificated.ndjson").exists():
        transform_data_to_ndjson(path_to_glob=path_to_glob, path_to_res=path_to_res)
    else:
        print("Data resource exists")

    cook = Cooking(
        infile="../data/raw/unificated.ndjson",
        voc_dir="../data/prepared/vocabularies", 
        tokenized_data="../data/prepared/tokenized_data.csv")
    cook.createVocabularies()

    print("\n======================================\n")

    cook.return_tokenized()

    seed = np.ones(7) # 1, 0.5, 10, 5, 0.1, 8, 0.2

    classification = Classification(
        infile="../data/prepared/tokenized_data.csv", 
        outfile="../data/prepared/tokenized_data_dot_product.csv",
        seed=seed
        )
    classification.calculate_dot_product()
