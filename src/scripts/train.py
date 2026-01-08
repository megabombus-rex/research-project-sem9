import argparse
from datetime import datetime
import torch

import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint

from src.scripts.model.ViT_multi_model_multilabel import ViTMonoMultilabelModel, ViTTabMultiModalMultilabelModel, ViTTextMultiModalMultilabelModel
from src.scripts.util.config_reader import load_models_config
from src.scripts.data.dataset import get_dataloaders

def vision_text_multilabel(batch_size: int = 32, num_workers: int = 4, epochs: int = 5, accelerator: str = 'cpu'):
    config = load_models_config()

    train_set, val_set, test_set = get_dataloaders(
        batch_size=batch_size, 
        num_workers=num_workers,
        mode="text", 
        tokenizer=config["models"]["vit-with-tokenizer"]["text-model"]
        )

    model = ViTTextMultiModalMultilabelModel(
        config["models"]["vit-with-tokenizer"], num_classes=14
    )
    
    now = datetime.now()
    f = now.strftime("%Y-%m-%d %H:%M:%S")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="data/models/text",
        filename="best_model-{epoch:02d}-{val_loss:.2f}" + f"-{f}"
    )
    
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        log_every_n_steps=1,
        deterministic=True,
        callbacks=[checkpoint_callback])
    
    trainer.fit(model, train_dataloaders=train_set, val_dataloaders=val_set)
    trainer.test(model, test_set)
    
def vision_tab_multilabel(batch_size: int = 32, num_workers: int = 4, epochs: int = 5, accelerator: str = 'cpu'):
    train_set, val_set, test_set = get_dataloaders(batch_size=batch_size, num_workers=num_workers, mode="tabular")
    config = load_models_config()

    model = ViTTabMultiModalMultilabelModel(
        config["models"]["vit-model-tabular"], num_classes=14
    )
    
    now = datetime.now()
    f = now.strftime("%Y-%m-%d %H:%M:%S")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="data/models/tab",
        filename="best_model-{epoch:02d}-{val_loss:.2f}" + f"-{f}"
    )
    
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        log_every_n_steps=1,
        deterministic=True,
        callbacks=[checkpoint_callback])
    
    trainer.fit(model, train_dataloaders=train_set, val_dataloaders=val_set)
    trainer.test(model, test_set)
    
def vision_mono_multilabel(batch_size: int = 32, num_workers: int = 4, epochs: int = 5, accelerator: str = 'cpu'):
    train_set, val_set, test_set = get_dataloaders(batch_size=batch_size, num_workers=num_workers, mode="tabular")
    config = load_models_config()

    model = ViTMonoMultilabelModel(
        config["models"]["vit-model-tabular"], num_classes=14
    )
    
    now = datetime.now()
    f = now.strftime("%Y-%m-%d %H:%M:%S")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="data/models/mono",
        filename="best_model-{epoch:02d}-{val_loss:.2f}" + f"-{f}"
    )
    
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        log_every_n_steps=1,
        deterministic=True,
        callbacks=[checkpoint_callback])
    
    trainer.fit(model, train_dataloaders=train_set, val_dataloaders=val_set)
    trainer.test(model, test_set)

if __name__ == "__main__":
    MAX_EPOCHS = 5
    BATCH_SIZE = 64
    ACCELERATOR = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 8
    LR = 1e-5
        
    L.seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--accelerator", type=str, default="auto")
    args = parser.parse_args()
    
    vision_text_multilabel(args.batch_size, num_workers=NUM_WORKERS, epochs=args.epochs, accelerator=args.accelerator)
    vision_tab_multilabel(args.batch_size, num_workers=NUM_WORKERS, epochs=args.epochs, accelerator=args.accelerator)
    vision_mono_multilabel(args.batch_size, num_workers=NUM_WORKERS, epochs=args.epochs, accelerator=args.accelerator)