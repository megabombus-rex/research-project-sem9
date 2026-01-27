import argparse
from datetime import datetime
import torch

import pytorch_lightning as L

from src.scripts.model.ViT_multi_model_multilabel import (
    ViTMonoMultilabelModel,
    ViTTabMultiModalMultilabelModel,
    ViTTextMultiModalMultilabelModel,
)
from src.scripts.util.config_reader import load_models_config
from src.scripts.data.dataset import get_dataloaders


def test(batch_size: int = 32,
         num_workers: int = 4,
         accelerator: str = 'cpu',
         ckpt_path: str = None,
         mode: str = 'text'):
    assert ckpt_path is not None, "You must provide --ckpt path to a .ckpt file"

    config = load_models_config()

    # Load dataloaders
    if mode == 'text':
        _, _, test_set = get_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
            mode="text",
            tokenizer=config["models"]["vit-with-tokenizer"]["text-model"],
        )
        model = ViTTextMultiModalMultilabelModel.load_from_checkpoint(
            ckpt_path,
            config=config["models"]["vit-with-tokenizer"],
            num_classes=14,
        )

    elif mode == 'tabular':
        _, _, test_set = get_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
            mode="tabular",
        )
        model = ViTTabMultiModalMultilabelModel.load_from_checkpoint(
            ckpt_path,
            config=config["models"]["vit-model-tabular"],
            num_classes=14,
        )

    elif mode == 'mono':
        _, _, test_set = get_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
            mode="tabular",
        )
        model = ViTMonoMultilabelModel.load_from_checkpoint(
            ckpt_path,
            config=config["models"]["vit-model-tabular"],
            num_classes=14,
        )
    else:
        raise ValueError("mode must be one of ['text', 'tabular', 'mono']")

    trainer = L.Trainer(
        accelerator=accelerator,
        deterministic=True,
        precision="16-mixed",
    )

    trainer.test(model, dataloaders=test_set)


if __name__ == "__main__":
    MAX_EPOCHS = 10
    BATCH_SIZE = 96
    ACCELERATOR = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 8

    L.seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--mode", type=str, required=True, choices=["text", "tabular", "mono"], help="Model type")

    args = parser.parse_args()

    test(
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        accelerator=args.accelerator,
        ckpt_path=args.ckpt,
        mode=args.mode,
    )
