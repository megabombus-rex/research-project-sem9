import json
import random
import time
import numpy as np
import torch
from tqdm import tqdm
from src.scripts.data.dataset import get_dataloaders
from src.scripts.experimental.wrappers.ViT_mono_model import ViTModel
from src.scripts.model.ViT_multi_model_multilabel import ViTMonoMultilabelModel, ViTTabMultiModalMultilabelModel, ViTTextMultiModalMultilabelModel
from src.scripts.util.config_reader import load_models_config
from src.scripts.experimental.wrappers.experiment_model import BaseModel


class WallClockTimeExperiment:
    def __init__(
        self,
        model: BaseModel,
        result_model_name: str,
        n_samples: int = 10000,
        iterations: int = 5,
        seed: int = 42
    ):
        self.model = model
        self.result_model_name = result_model_name
        self.n_samples = n_samples
        self.iterations = iterations
        self.seed = seed
        self.rng = random.Random(seed)

    def run(self, data_loader, device = None):
        mean_times = np.zeros((self.iterations))
        synchronize = False 
        
        if device is not None and device != 'cpu':
            self.model.model.to(device)
            if device == "cuda":
                synchronize = True
        
        self.model.model.eval()

        for i in tqdm(range(self.iterations)):
            n_seen = 0
            if synchronize:
                torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad():
                for batch in data_loader:
                    if device is not None and device != 'cpu':
                        batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
                    _ = self.model.predict(batch)
                    n_seen += 1
                    if n_seen >= self.n_samples:
                        break
            
            if synchronize:
                torch.cuda.synchronize()
            end_time = time.time()
            mean_times[i] = (end_time - start_time) / self.n_samples

        mean = np.mean(mean_times)
        std = np.std(mean_times)
        print(
            f"After {self.iterations} iterations the mean time for each iteration was:"
        )
        for i in range(self.iterations):
            print(f"{i+1}: {mean_times[i]} per sample.")
        print(f"Mean time across all iterations is {mean}({std}) per sample .")

        data = {
            'model': self.result_model_name,
            'mean_times': mean_times.tolist(),
            'iterations': self.iterations,
            'mean_time_per_sample': mean,
            'std_per_sample': std
        }

        with open(f'{self.result_model_name}.json', "w") as f:
            json.dump(data, f)
        
if __name__ == '__main__':
    config = load_models_config()
    BATCH_SIZE = 1
    NUM_WORKERS = 0
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {DEVICE}')
    
    N_SAMPLES = 1000
    ITERATIONS = 10
    SEED = 42

    print('========== WCT Experiment Monomodal ViT ==========')

    _, _, test_set = get_dataloaders(
        train_ratio=0.1,
        val_ratio = 0.1,
        test_ratio=0.8,
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        mode="mono"
        )
    
    vit_mono = ViTMonoMultilabelModel(
        config["models"]["vit-model-tabular"], num_classes=14
    )
    monoVisionModel = ViTModel(vit_mono)
    
    exp = WallClockTimeExperiment(model=monoVisionModel, result_model_name='ViT', n_samples=N_SAMPLES, iterations=ITERATIONS, seed=SEED)
    exp.run(test_set, device=DEVICE)
    
    print('========== WCT Experiment ViT + TabTransformer ==========')
    
    _, _, test_set = get_dataloaders(
        train_ratio=0.1,
        val_ratio = 0.1,
        test_ratio=0.8,
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        mode="tabular"
        )
    
    vit_tab = ViTTabMultiModalMultilabelModel(
        config["models"]["vit-model-tabular"], num_classes=14
    )
    multiTabVisionModel = ViTModel(vit_tab)
    
    exp = WallClockTimeExperiment(model=multiTabVisionModel, result_model_name='ViT_TabTransformer', n_samples=N_SAMPLES, iterations=ITERATIONS, seed=SEED)
    exp.run(test_set, device=DEVICE)
    
    print('========== WCT Experiment ViT + BERT ==========')
    
    _, _, test_set = get_dataloaders(
        train_ratio=0.1,
        val_ratio = 0.1,
        test_ratio=0.8,
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        mode="text", 
        tokenizer=config["models"]["vit-with-tokenizer"]["text-model"]
        )
    
    vit_text = ViTTextMultiModalMultilabelModel(
        config["models"]["vit-with-tokenizer"], num_classes=14
    )
    multiTextVisionModel = ViTModel(vit_text)
    
    exp = WallClockTimeExperiment(model=multiTextVisionModel, result_model_name='ViT_BERT', n_samples=N_SAMPLES, iterations=ITERATIONS, seed=SEED)
    exp.run(test_set, device=DEVICE)
