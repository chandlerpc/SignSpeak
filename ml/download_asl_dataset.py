#!/usr/bin/env python3
"""Download ASL Alphabet dataset from Kaggle"""
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Create data directory
data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

print("Downloading ASL Alphabet dataset from Kaggle...")
print("Dataset: grassknoted/asl-alphabet")
print(f"Download location: {data_dir}")
print("\nThis is a large dataset (~1GB). Download may take several minutes...\n")

# Download the dataset
api.dataset_download_files(
    'grassknoted/asl-alphabet',
    path=data_dir,
    unzip=True
)

print("\nDataset downloaded and extracted successfully!")
print(f"Dataset location: {os.path.abspath(data_dir)}")

# List contents
print("\nDataset contents:")
for item in os.listdir(data_dir):
    item_path = os.path.join(data_dir, item)
    if os.path.isdir(item_path):
        num_files = len(os.listdir(item_path)) if os.path.isdir(item_path) else 0
        print(f"  - {item}/ ({num_files} items)")
    else:
        size_mb = os.path.getsize(item_path) / (1024 * 1024)
        print(f"  - {item} ({size_mb:.2f} MB)")
