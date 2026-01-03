# this should be a combination of a Vision Transformer (ViT) and a text embedding transformer with some sort of a fusion layer

from sklearn.metrics import precision_score, f1_score, recall_score
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForImageClassification, AutoTokenizer

class FusionModule(nn.Module):
    def __init__(self, in_size, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_size, in_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_size // 2, num_classes)
        )

    def forward(self, img_embeds, txt_embeds):
        fused = torch.cat([img_embeds, txt_embeds], dim=-1)
        return self.fc(fused)
    
class ViTMultiModalModel(nn.Module):
    def __init__(self, config, num_classes: int, lr: float = 0.001):
        super().__init__(self)
        self.vision_model_name = config['image-model']
        self.text_model_name = config['text-model']
        self.tokenizer_name = config['text-model']
        self.vision_model = AutoModelForImageClassification.from_pretrained(self.vision_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.text_model = AutoModel.from_pretrained(self.text_model_name)
        self.fusion = FusionModule(self.vision_model.hidden_size + self.text_model.hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.num_classes = num_classes
        
    def forward(self, x):
        pixel_values = x['pixel_values']
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        
        outputs_v = self.vision_model(pixel_values)
        outputs_t = self.text_model(input_ids, attention_mask)
        img_embeds = outputs_v.last_hidden_state[:, 0]
        txt_embeds = outputs_t.last_hidden_state[:, 0]
        
        logits = self.fusion(img_embeds, txt_embeds)
        return logits
        
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch, train_batch["labels"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        f1 = f1_score(preds, y, task="multiclass", num_classes=self.num_classes)
        prec_score = precision_score(preds, y, task="multiclass", num_classes=self.num_classes)
        rec_score = recall_score(preds, y, task="multiclass", num_classes=self.num_classes)
        print(f'Training - Accuracy: {acc}, F1: {f1}, Precision: {prec_score}, Recall: {rec_score}')
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch, val_batch["labels"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        f1 = f1_score(preds, y, task="multiclass", num_classes=self.num_classes)
        prec_score = precision_score(preds, y, task="multiclass", num_classes=self.num_classes)
        rec_score = recall_score(preds, y, task="multiclass", num_classes=self.num_classes)
        print(f'Validation - Accuracy: {acc}, F1: {f1}, Precision: {prec_score}, Recall: {rec_score}')
        return loss

    def predict_step(self, test_batch, batch_idx):
        x, y = test_batch, test_batch["labels"]
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return {"preds": preds, "targets": y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)