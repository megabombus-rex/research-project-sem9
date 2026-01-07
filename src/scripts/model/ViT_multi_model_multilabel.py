import pytorch_lightning as L
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModel, AutoTokenizer

from src.scripts.model.fusion_module import FusionModule

class ViTTextMultiModalMultilabelModel(L.LightningModule):
    def __init__(
        self, 
        config, 
        num_classes: int, 
        lr: float = 1e-3, 
        optimizer: str = "SGD"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr
        self.optimizer_name = optimizer
        
        self.vision_model_name = config["image-model"]
        self.text_model_name = config["text-model"]
        self.tokenizer_name = config["text-model"]

        self.vision_model       = AutoModel.from_pretrained(self.vision_model_name)
        self.tokenizer          = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.text_model         = AutoModel.from_pretrained(self.text_model_name)
        vision_dim, text_dim    = self.vision_model.config.hidden_size, self.text_model.config.hidden_size
        self.fusion             = FusionModule(vision_dim + text_dim, num_classes)

        for p in self.vision_model.parameters():
            p.requires_grad = False

        for p in self.text_model.parameters():
            p.requires_grad = False

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        pixel_values = x["pixel_values"]
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]

        outputs_v = self.vision_model(pixel_values)
        outputs_t = self.text_model(input_ids, attention_mask)
        img_embeds = outputs_v.last_hidden_state[:, 0]
        txt_embeds = outputs_t.last_hidden_state[:, 0]

        logits = self.fusion(img_embeds, txt_embeds)
        return logits
    
    def training_step(self, batch, batch_idx):
        images, metadata, labels = batch
        transformed_text = self.tokenizer(
            metadata,
            padding=True,
            truncation=True,
            return_tensors="pt"
            )
        
        x = {
            "pixel_values": images,
            "input_ids": torch.tensor(transformed_text["input_ids"]).to(self.device),
            "attention_mask": torch.tensor(transformed_text["attention_mask"]).to(self.device)
        }
        logits = self(x)
        loss = self.loss_fn(logits, labels.float())
        self.log("train_loss", loss, prog_bar=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        images, metadata, labels = batch
        transformed_text = self.tokenizer(
            metadata,
            padding=True,
            truncation=True,
            return_tensors="pt"
            )
        
        x = {
            "pixel_values": images,
            "input_ids": torch.tensor(transformed_text["input_ids"]).to(self.device),
            "attention_mask": torch.tensor(transformed_text["attention_mask"]).to(self.device)
        }
        logits = self(x)
        loss = self.loss_fn(logits, labels.float())
        self.log("val_loss", loss, prog_bar=True, batch_size=images.size(0))
        return loss

    def test_step(self, batch, batch_idx):
        images, metadata, labels = batch
        transformed_text = self.tokenizer(
            metadata,
            padding=True,
            truncation=True,
            return_tensors="pt"
            )
        
        x = {
            "pixel_values": images,
            "input_ids": torch.tensor(transformed_text["input_ids"]).to(self.device),
            "attention_mask": torch.tensor(transformed_text["attention_mask"]).to(self.device)
        }
        logits = self(x)
        loss = self.loss_fn(logits, labels.float())
        self.log("test_loss", loss, batch_size=images.size(0))
        return loss

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        if self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer
    
class ViTTabMultiModalMultilabelModel(L.LightningModule):
    def __init__(
        self,
        config,
        num_classes: int,
        lr: float = 1e-3,
        optimizer: str = "SGD"
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.lr = lr
        self.optimizer_name = optimizer

        self.vision_model_name = config["image-model"]
        self.tabular_model_config = config["tab-model"]

        self.vision_model = AutoModel.from_pretrained(self.vision_model_name)
        self.tabular_model = TabularTransformerModel(self.tabular_model_config)

        vision_dim = self.vision_model.config.hidden_size
        tab_dim = self.tabular_model.embed_dim

        self.fusion = FusionModule(vision_dim + tab_dim, num_classes)

        # due to multilabel logic - this should be the loss
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        pixel_values = x["pixel_values"]
        tabular_values = x["tabular_values"]


        outputs_v = self.vision_model(pixel_values)
        img_embeds = outputs_v.last_hidden_state[:, 0]

        tab_embeds = self.tabular_model(tabular_values)

        logits = self.fusion(img_embeds, tab_embeds)
        return logits

    def training_step(self, batch, batch_idx):
        images, metadata, labels = batch
        x = {
            "pixel_values": images,
            "tabular_values": metadata
        }
        logits = self(x)
        loss = self.loss_fn(logits, labels.float())
        self.log("train_loss", loss, prog_bar=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        images, metadata, labels = batch
        x = {
            "pixel_values": images,
            "tabular_values": metadata
        }
        logits = self(x)
        loss = self.loss_fn(logits, labels.float())
        self.log("val_loss", loss, prog_bar=True, batch_size=images.size(0))

    def test_step(self, batch, batch_idx):
        images, metadata, labels = batch
        x = {
            "pixel_values": images,
            "tabular_values": metadata
        }
        logits = self(x)
        loss = self.loss_fn(logits, labels.float())
        self.log("test_loss", loss, batch_size=images.size(0))

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        if self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer
        
class TabularTransformerModel(nn.Module):    
    def __init__(self, model_config):
        super().__init__()
        embed_dim = model_config.get("embed_dim", 64)
        depth = model_config.get("depth", 3)
        num_heads = model_config.get("num_heads", 4)
        dropout = model_config.get("dropout", 0.1)
        self.lr = model_config.get("learning_rate", 1e-4)
        self.embed_dim = embed_dim
        
        # Feature embedding
        self.embed = nn.Linear(1, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True
            )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classifier head - not needed
        # self.classifier = nn.Linear(embed_dim, 2)
        # self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = self.embed(x.unsqueeze(-1))  # (batch, features, embed_dim)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        return x[:, 0]
    
class ViTMonoMultilabelModel(L.LightningModule):
    def __init__(
        self,
        config,
        num_classes: int,
        lr: float = 1e-3,
        optimizer: str = "SGD"
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.lr = lr
        self.optimizer_name = optimizer

        self.vision_model_name = config["image-model"]

        self.vision_model = AutoModel.from_pretrained(self.vision_model_name)
        in_size = self.vision_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(in_size, in_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_size // 2, num_classes),
        )

        # due to multilabel logic - this should be the loss
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        outputs_v = self.vision_model(x["pixel_values"])
        logits = self.classifier(outputs_v.last_hidden_state[:, 0])
        return logits
    

    def training_step(self, batch, batch_idx):
        images, _, labels = batch
        x = {
            "pixel_values": images
        }
        logits = self(x)
        loss = self.loss_fn(logits, labels.float())
        self.log("train_loss", loss, prog_bar=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        images, _, labels = batch
        x = {
            "pixel_values": images
        }
        logits = self(x)
        loss = self.loss_fn(logits, labels.float())
        self.log("val_loss", loss, prog_bar=True, batch_size=images.size(0))

    def test_step(self, batch, batch_idx):
        images, _, labels = batch
        x = {
            "pixel_values": images
        }
        logits = self(x)
        loss = self.loss_fn(logits, labels.float())
        self.log("test_loss", loss, batch_size=images.size(0))

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        if self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer