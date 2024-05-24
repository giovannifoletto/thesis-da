# Thesis Data Analysis

> Giovanni Foletto - Jan. 29, 2024

This repository contains the evaluation, preparation and analysis on the data for the thesis.

The data initial works is done on the [Flaws Event Log Dump](https://summitroute.com/blog/2020/10/09/public_dataset_of_cloudtrail_logs_from_flaws_cloud/).

All the data repository data is done with `dvc`.


## Get The datasets

All the dataset are stored with DVC inside GDRIVE.

To enable DVC, you need a new env:

```bash
$ python3 -m venv .; source ./bin/activate
```

And then, if all is gone right, we should install all the needed dependencies.
```bash
(env-name) $ pip install -r requirements.txt
(env-name) $ dvc pull
```

The dataset are (quite) big, if you like to download only one of them, you can try:

```bash
(env-name) $ dvc fetch <name-of-the-dataset>
```

To find the name of the dataset, you can use the `.dvc` files in the repo. The name of the resource are the same but without `.dvc` at the end.

