import torch
from torch.utils.data import DataLoader, TensorDataset

# dummy data
image_data = torch.randn(1000, 3, 224, 224) 
labels = torch.randint(0, 10, (1000,))  

dataset = TensorDataset(image_data, labels)

#Split into batches
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#to view every iterated batch
for batch_images, batch_labels in dataloader:
    print(f"Batch shape: {batch_images.shape}, Labels: {batch_labels}")