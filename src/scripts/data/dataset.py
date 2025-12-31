import os
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from torchvision import transforms
from typing import List, Tuple
from collections import Counter

from download_dataset import download_medtrinity_dataset

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(script_dir, "../../../data/MedTrinity-25M"))

if not os.path.exists(data_dir):
    ds = download_medtrinity_dataset(data_dir)
else:
    ds = load_from_disk(data_dir)

class BasicTokenizer:
    def __init__(self, captions: List[str], max_vocab_size: int = 10000, min_freq: int = 2):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        
        counter = Counter()
        for caption in captions:
            counter.update(self._tokenize(caption))
        
        vocab = [self.pad_token, self.unk_token]
        common_words = [w for w, f in counter.most_common(max_vocab_size) if f >= min_freq]
        vocab.extend(common_words)
        
        self.word2idx = {word: i for i, word in enumerate(vocab)}
        self.idx2word = {i: word for i, word in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def __call__(self, text: str, max_length: int, padding: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self._tokenize(text)[:max_length]
        input_ids = [self.word2idx.get(token, self.word2idx[self.unk_token]) for token in tokens]
        
        attention_mask = [1] * len(input_ids)
        
        if padding and len(input_ids) < max_length:
            diff = max_length - len(input_ids)
            input_ids.extend([self.word2idx[self.pad_token]] * diff)
            attention_mask.extend([0] * diff)
            
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)


class MedTrinityDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, transform=None, max_length=77):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.dataset[idx]
        
        image = item['image'].convert("RGB")
        image = self.transform(image)
        
        input_ids, attention_mask = self.tokenizer(
            item["caption"],
            max_length=self.max_length
        )
        
        return image, input_ids, attention_mask

def get_dataloaders(batch_size: int = 32, num_workers: int = 4):
    #70% train, 15% test, 15% val
    train_test_split = ds['train'].train_test_split(test_size=0.3, seed=42)
    test_val_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)
    
    train_data = train_test_split['train']
    
    all_captions = train_data["caption"]
    tokenizer = BasicTokenizer(all_captions)
    print('tokenizer.vocab_size: ', tokenizer.vocab_size)

    train_set = MedTrinityDataset(train_data, tokenizer=tokenizer)
    val_set   = MedTrinityDataset(test_val_split['test'], tokenizer=tokenizer)
    test_set  = MedTrinityDataset(test_val_split['train'], tokenizer=tokenizer)

    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
    }

    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader   = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader  = DataLoader(test_set, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    
    for images, input_ids, attention_masks in train_loader:
        print(f"Images shape: {images.shape}")
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Attention Mask shape: {attention_masks.shape}")
        break
