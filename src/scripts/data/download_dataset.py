import os
from datasets import load_dataset


def download_medtrinity_dataset(output_dir):
    print(f"Downloading MedTrinity-25M demo dataset...")
    ds = load_dataset("UCSC-VLAA/MedTrinity-25M", name="25M_demo")
    
    print("Dataset loaded")
    print(f"Dataset structure: {ds}")
    
    os.makedirs(output_dir, exist_ok=True)    
    ds.save_to_disk(output_dir)
    
    print(f"Dataset saved to {output_dir}")
    return ds


if __name__ == "__main__":
    download_medtrinity_dataset()