import os
import mlcroissant as mlc
import pandas as pd
from datasets import Dataset


def download_chexpert_dataset(output_dir=None):
    print(f"Downloading demo dataset...")
    
    croissant_dataset = mlc.Dataset('https://www.kaggle.com/datasets/ashery/chexpert/croissant/download')
    
    record_sets = croissant_dataset.metadata.record_sets
    print(f"Record sets: {record_sets}")
    
    record_set_df = pd.DataFrame(croissant_dataset.records(record_set=record_sets[0].uuid))
    print(f"Dataset shape: {record_set_df.shape}")
    print(f"Dataset columns: {record_set_df.columns.tolist()}")
    print(f"First example:\n{record_set_df.head()}")
    
    ds = Dataset.from_pandas(record_set_df)
    
    os.makedirs(output_dir, exist_ok=True)    
    ds.save_to_disk(output_dir)
    
    print(f"Dataset saved to {output_dir}")
    return ds


if __name__ == "__main__":
    download_chexpert_dataset()