import json
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from iterstrat.ml_stratifiers import RepeatedMultilabelStratifiedKFold

import torch
from torch.utils.data import Subset, DataLoader

import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint

from src.scripts.data.dataset import get_dataset
from src.scripts.experimental.wrappers.ViT_mono_model import ViTModel
from src.scripts.model.ViT_multi_model_multilabel import ViTMonoMultilabelModel, ViTTabMultiModalMultilabelModel, ViTTextMultiModalMultilabelModel
from src.scripts.util.config_reader import load_models_config
from src.scripts.experimental.wrappers.experiment_model import BaseModel

BATCH_SIZE = 32
NUM_WORKERS = 4

PATHOLOGY_COLUMNS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]

class KFoldExperiment:
    def __init__(
        self, config, mode: str, model: BaseModel, n_folds: int = 5, n_repeats: int = 2, seed: int = 42, max_epochs: int = 5
    ):
        self.config = config
        self.model = model
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.max_epochs = max_epochs
        self.seed = seed
        self.mode = mode

    def run(self, device = None):
        print(f'Config: {self.config}')
        dataset = get_dataset(mode=self.mode, tokenizer=self.config.get('text-model', None))
        X = dataset.df.drop(columns=PATHOLOGY_COLUMNS)
        y = dataset.df[PATHOLOGY_COLUMNS].values

        if device is not None and device != 'cpu':
            self.model.model.to(device)

        rmskf = RepeatedMultilabelStratifiedKFold(
            n_splits=self.n_folds, n_repeats=self.n_repeats, random_state=self.seed
        )
        model_type = type(self.model.model)
        model_str = str(model_type).replace('class', '').replace('src.scripts.model.', '')
        checkpoint_callback = ModelCheckpoint(
            monitor="train_loss",
            dirpath="data/models/kfold",
            filename="best_model-{epoch:02d}-{train_loss:.2f}" + f"-{model_str}"
        )
        
        # n_folds * n_repeats x single dataset/single model -> (n_folds * n_repeats)
        losses = np.zeros(self.n_folds * self.n_repeats)
        AUROCs = np.zeros(self.n_folds * self.n_repeats)
        mAPs_micro = np.zeros(self.n_folds * self.n_repeats)
        mAPs_macro = np.zeros(self.n_folds * self.n_repeats)
        F1s = np.zeros(self.n_folds * self.n_repeats)

        for i, (train_idx, test_idx) in enumerate(rmskf.split(X, y)):
            model = model_type(self.config, num_classes=14)
            train_set = Subset(dataset, train_idx)
            test_set = Subset(dataset, test_idx)

            loader_args = {
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "pin_memory": True,
            }

            if self.mode == "text":
                loader_args["collate_fn"] = dataset.text_collate_fn

            train_loader = DataLoader(train_set, shuffle=True, **loader_args)
            test_loader = DataLoader(test_set, shuffle=False, **loader_args)
            
            trainer = L.Trainer(
                max_epochs=self.max_epochs,
                accelerator= 'cuda' if torch.cuda.is_available() else 'cpu',
                log_every_n_steps=1,
                deterministic=True,
                callbacks=[checkpoint_callback])
            
            trainer.fit(model, train_dataloaders=train_loader)
            result = trainer.test(model, test_loader)
            print(f'Result: {result}')
            losses[i] = result[0]['test_loss']
            AUROCs[i] = result[0]['test_auroc']
            mAPs_micro[i] = result[0]['test_map']
            mAPs_macro[i] = result[0]['test_map_micro']
            F1s[i] = result[0]['test_f1']
        
        data = {
            'model': model_str,
            'auroc': AUROCs.tolist(),
            'mAP-micro': mAPs_micro.tolist(),
            'mAP-macro': mAPs_macro.tolist(),
            'F1': F1s.tolist()
        }

        with open(f'{model_str}.json', "w") as f:
            json.dump(data, f)

        


if __name__ == '__main__':
    config = load_models_config()
    N_FOLDS = 5
    N_REPEATS = 2
    SEED = 42
    MAX_EPOCHS = 1

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {DEVICE}')

    print('========== K-Fold Experiment Monomodal ViT ==========')
    vit_mono = ViTMonoMultilabelModel(
        config["models"]["vit-model-tabular"], num_classes=14
    )
    monoVisionModel = ViTModel(vit_mono)
    print(f'Config passed to the model: {config["models"]["vit-model-tabular"]}')

    exp = KFoldExperiment(config=config["models"]["vit-model-tabular"], mode="mono", model=monoVisionModel, n_folds=N_FOLDS, n_repeats=N_REPEATS, seed=SEED, max_epochs=MAX_EPOCHS)
    exp.run(device=DEVICE)
    
    print('========== K-Fold Experiment ViT + TabTransformer ==========')
    vit_tab = ViTTabMultiModalMultilabelModel(
        config["models"]["vit-model-tabular"], num_classes=14
    )
    multiTabVisionModel = ViTModel(vit_tab)

    exp = KFoldExperiment(config=config["models"]["vit-model-tabular"], mode="tabular", model=multiTabVisionModel, n_folds=N_FOLDS, n_repeats=N_REPEATS, seed=SEED, max_epochs=MAX_EPOCHS)
    exp.run(device=DEVICE)

    print('========== K-Fold Experiment ViT + BERT ==========')
    vit_text = ViTTextMultiModalMultilabelModel(
        config["models"]["vit-with-tokenizer"], num_classes=14
    )
    multiTextVisionModel = ViTModel(vit_text)

    exp = KFoldExperiment(config=config["models"]["vit-with-tokenizer"], mode="text", model=multiTextVisionModel, n_folds=N_FOLDS, n_repeats=N_REPEATS, seed=SEED, max_epochs=MAX_EPOCHS)
    exp.run(device=DEVICE)