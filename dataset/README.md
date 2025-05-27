# Landscape Pictures Dataset Processing

This README provides instructions for downloading, extracting, and processing the landscape pictures dataset from Kaggle.

## Dataset Source

The dataset is sourced from Kaggle: Landscape Pictures by Arnaud58 [![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/datasets/arnaud58/landscape-pictures)
. Follow this link: [Kaggle Dataset](https://www.kaggle.com/datasets/arnaud58/landscape-pictures)

## Setup

1. **Create a Dataset Directory**: Create a directory to store the dataset:

```python
import os

ds_path = "./dataset/landscape-pictures"
os.makedirs(ds_path, exist_ok=True)
```

2. **Download the Dataset**: Use the following command to download the dataset from Kaggle:

```bash
curl -L https://www.kaggle.com/api/v1/datasets/download/arnaud58/landscape-pictures -o ./dataset/landscape-pictures.zip
```

Note: You may need a Kaggle API token for authentication. Ensure you have the `kaggle.json` file configured in `~/.kaggle/` or set up the Kaggle API as per Kaggle's API documentation.

3. **Extract the Dataset**: Run the following Python code to extract the downloaded zip file:

```python
import zipfile
import os

with zipfile.ZipFile('dataset/landscape-pictures.zip', 'r') as zip_ref:
    zip_ref.extractall(ds_path)
```

This will extract the dataset into the `./dataset` directory.
