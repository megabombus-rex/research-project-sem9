import torch
from tqdm import tqdm

from src.scripts.model.ViT_multi_model import ViTTextMultiModalModel, ViTTabMultiModalMutilabelModel
from src.scripts.util.config_reader import load_models_config
from src.scripts.data.dataset import get_dataloaders

class Trainer():
    def __init__(self, max_epochs: int = 50, batch_size: int = 32, device: str = 'cuda'):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    def fit(self, model, train_loader, test_loader=None, val_loader=None):
        model.to(self.device)
        
        for _ in tqdm(range(self.max_epochs)):
            train_loss = self._train_epoch(model, train_loader)
            val_loss = None

            if val_loader is not None:
                val_loss = self._val_epoch(model, val_loader)
            
            print(f"Train loss: {train_loss:.4f}" + (f" | Val loss: {val_loss:.4f}" if val_loss else ""))
        
        if test_loader is not None:
            self._test(model, test_loader)
        
    def _train_epoch(self, model, loader):
        model.train()
        total_loss = 0.0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            model.optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            model.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)
    
    def _val_epoch(self, model, loader):
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = model.compute_loss(batch)
                total_loss += loss.item()

        return total_loss / len(loader)

    def _test(self, model, loader):
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = model.compute_loss(batch)
                total_loss += loss.item()

        print(f"Test loss: {total_loss / len(loader):.4f}")
        
if __name__ == '__main__':
    MAX_EPOCHS = 5
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = load_models_config()
    
    torch.manual_seed(42)
    
    train_set, val_set, test_set, labels = get_dataloaders()
    num_classes = len(labels)
    
    # model = ViTTextMultiModalModel(config['models']['vit-with-tokenizer'], num_classes=num_classes)
    model = ViTTabMultiModalMutilabelModel(config['models']['vit-model-tabular'], num_classes=num_classes)
    
    trainer = Trainer(MAX_EPOCHS, BATCH_SIZE, DEVICE)
    trainer.fit(model, train_set, test_set, val_set)
    