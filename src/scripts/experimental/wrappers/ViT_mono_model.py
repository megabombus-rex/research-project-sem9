import pytorch_lightning as L
import torch

# from src.scripts.model.ViT_multi_model_multilabel import ViTMonoMultilabelModel
from src.scripts.experimental.wrappers.experiment_model import BaseModel


class ViTModel(BaseModel):
    def __init__(self, model: L.LightningModule):
        super().__init__(model)
    
    def __call__(self, x):
        return self.model(x)
    
    def predict(self, x):
        with torch.no_grad():
            pred = self.model.predict_step(x)
        
        return pred        