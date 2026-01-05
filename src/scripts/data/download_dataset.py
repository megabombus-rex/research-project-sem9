import os
import zipfile
import requests
import shutil
from pathlib import Path
from tqdm import tqdm


def download_chexpert_dataset(output_dir="data/chexpert"):
    print("Downloading CheXpert dataset")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    url = "https://www.kaggle.com/api/v1/datasets/download/ashery/chexpert"
    zip_path = output_path / "chexpert.zip"
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    temp_dir = output_path / "temp_extract"
    temp_dir.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    chexpert_dir = output_path / "CheXpert-v1.0-small"
    chexpert_dir.mkdir(exist_ok=True)
    
    for item in temp_dir.iterdir():
        dest = chexpert_dir / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(item), str(dest))
    
    temp_dir.rmdir()
    zip_path.unlink()
    
    print(f"Dataset downloaded to {chexpert_dir}")
    return output_path


if __name__ == "__main__":
    download_chexpert_dataset()