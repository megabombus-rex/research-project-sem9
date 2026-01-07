import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Tuple, Literal
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from download_dataset import download_chexpert_dataset

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(script_dir, "../../../data/chexpert/CheXpert-v1.0-small"))

if not os.path.exists(data_dir):
    download_chexpert_dataset(os.path.dirname(data_dir))


def text_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    texts = [item[1] for item in batch]
    labels = torch.stack([item[2] for item in batch])
    return images, texts, labels


class CheXpertDataset(Dataset):
    METADATA_COLUMNS = ['Sex', 'Age', 'Frontal/Lateral', 'AP/PA']
    
    PATHOLOGY_COLUMNS = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices'
    ]
    
    def __init__(
        self,
        data_dir: str,
        mode: Literal["tabular", "text"] = "tabular",
        transform=None,
        u_zeros: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.u_zeros = u_zeros
        
        train_csv = self.data_dir / "train.csv"
        valid_csv = self.data_dir / "valid.csv"
        
        dfs = []
        if train_csv.exists():
            dfs.append(pd.read_csv(train_csv))
        if valid_csv.exists():
            dfs.append(pd.read_csv(valid_csv))
        
        if not dfs:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        self.df = pd.concat(dfs, ignore_index=True)
        
        for col in self.PATHOLOGY_COLUMNS:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0.0)
                if self.u_zeros:
                    self.df[col] = self.df[col].replace(-1.0, 0.0)
        
        if self.mode == "tabular":
            self._prepare_encoders()
        
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded {len(self.df)} samples")
    
    def _prepare_encoders(self):
        self.sex_encoder = LabelEncoder()
        self.sex_encoder.fit(self.df['Sex'].fillna('Unknown'))
        
        self.view_encoder = LabelEncoder()
        self.view_encoder.fit(self.df['Frontal/Lateral'].fillna('Unknown'))
        
        self.position_encoder = LabelEncoder()
        self.position_encoder.fit(self.df['AP/PA'].fillna('Unknown'))
        
        self.age_scaler = MinMaxScaler()
        ages = self.df['Age'].fillna(self.df['Age'].median()).values.reshape(-1, 1)
        self.age_scaler.fit(ages)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _get_metadata_tabular(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        sex = self.sex_encoder.transform([row.get('Sex', 'Unknown')])[0]
        age = self.age_scaler.transform([[row.get('Age', self.df['Age'].median())]])[0][0]
        view = self.view_encoder.transform([row.get('Frontal/Lateral', 'Unknown')])[0]
        position = self.position_encoder.transform([row.get('AP/PA', 'Unknown')])[0]
        
        return torch.tensor([sex, age, view, position], dtype=torch.float32)
    
    def _get_metadata_text(self, idx: int) -> str:
        row = self.df.iloc[idx]
        sex = row.get('Sex', 'Unknown')
        age = row.get('Age', 'Unknown')
        view = row.get('Frontal/Lateral', 'Unknown')
        position = row.get('AP/PA', 'Unknown')
        
        return f"Patient: {sex}, Age {age}. X-ray view: {view} {position}."
    
    def _get_labels(self, idx: int) -> torch.Tensor:
        """Get pathology labels as tensor."""
        row = self.df.iloc[idx]
        labels = [float(row.get(col, 0.0)) for col in self.PATHOLOGY_COLUMNS]
        return torch.tensor(labels, dtype=torch.float32)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor | str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        img_path_str = row['Path']
        if img_path_str.startswith("CheXpert-v1.0-small/"):
            img_path_str = img_path_str[len("CheXpert-v1.0-small/"):]
        
        img_path = self.data_dir / img_path_str
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        if self.mode == "tabular":
            metadata = self._get_metadata_tabular(idx)
        else:
            metadata = self._get_metadata_text(idx)
        
        labels = self._get_labels(idx)
        
        return image, metadata, labels


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    mode: Literal["tabular", "text"] = "tabular",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    full_dataset = CheXpertDataset(data_dir, mode=mode)
    
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    
    if mode == "text":
        loader_args["collate_fn"] = text_collate_fn
    
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("TABULAR MODE")
    train_loader, _, _ = get_dataloaders(batch_size=4, mode="tabular")
    
    for images, metadata, labels in train_loader:
        print(f"Images: {images.shape}")
        print(f"Metadata: {metadata.shape}")
        print(f"Labels: {labels.shape}")
        print(f"Sample metadata: {metadata[0]}")
        print(f"Sample labels: {labels[0]}")
        break
    
    print("TEXT MODE")
    train_loader, _, _ = get_dataloaders(batch_size=4, mode="text")
    
    for images, metadata_texts, labels in train_loader:
        print(f"Images: {images.shape}")
        print(f"Metadata text: {metadata_texts[0]}")
        print(f"Labels: {labels.shape}")
        print(f"Sample labels: {labels[0]}")
        break