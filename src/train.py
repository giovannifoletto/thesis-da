#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset, DataLoader

import IPython

import polars as pl
from pathlib import Path
import math

dataset_location = Path("../data/prepared/tokenized_data.csv")


option = dict()
option["data_train"] = 0.6
option["data_test"] = 0.4

class FlawsCloudtrail(Dataset):
    def __init__(self) -> None:
        super().__init__()
        assert dataset_location.exists()

        df = pl.read_csv(dataset_location)
        self.df = torch.tensor(df.to_numpy())
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        return self.df[index]

if __name__ == "__main__":

    train_data = FlawsCloudtrail()
    test_data = FlawsCloudtrail()

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
