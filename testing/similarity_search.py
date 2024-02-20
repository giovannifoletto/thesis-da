import numpy as np
import polars as pl
import pickle
import random
from tqdm import tqdm

## 100% precision: [0.09848446, 0.52740314, 0.83209034, 0.00339979, 0.87988932, 0.30239244, 0.02079325]

dlp = pl.read_csv("../data/prepared/tokenized_data.csv")
equality_rate = dlp.unique().shape[0]

dir_name = "../data/prepared/sim_search"

seed = np.ones(dlp.shape[1])
best_seed = []
how_change = np.ones(dlp.shape[1])
changing_value = 0.
latest_change = 0.
epochs = 300
for i in tqdm(range(epochs)):
    x_train = []
    print(seed)
    for line in dlp.to_numpy():
        x_train.append(np.dot(seed, line))

    # evaluation
    num_after = pl.DataFrame(x_train).unique().shape[0]
    change = num_after/equality_rate

    if change > latest_change :
        best_seed = seed
    
    seed = np.random.rand(seed.shape[0])
    
    with open(f"{dir_name}/{i}_train_np_array.npy", "wb") as f:
        np.save(f, np.array(x_train))

    with open(f"{dir_name}/{i}_train_np_seed.txt", "w") as f:
        for num in seed:
            f.write(f"{num}, ")
        f.write("\n") 