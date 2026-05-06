import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pytorch_lightning as L
import torch
import torch.nn as nn
from torchmetrics.classification import MultilabelConfusionMatrix
from transformers import AutoModel, AutoModel, AutoTokenizer
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision, MultilabelF1Score
import torchvision.models as models

from src.scripts.util.misc import PATHOLOGY_COLUMNS
from src.scripts.model.fusion_module import FusionModule

class ResnetTextMultiModalMultilabelModel(L.LightningModule):
    def __init__(
        self, 
        config, 
        num_classes: int, 
        prediction_threshold: float = 0.5,
        lr: float = 1e-3, 
        optimizer: str = "AdamW"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr
        self.optimizer_name = optimizer
        
        self.vision_model_name = config["image-model"]
        self.text_model_name = config["text-model"]
        self.tokenizer_name = config["text-model"]

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        vision_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()           

        self.vision_model       = resnet
        self.tokenizer          = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.text_model         = AutoModel.from_pretrained(self.text_model_name)
        text_dim                = self.text_model.config.hidden_size
        self.fusion             = FusionModule(vision_dim + text_dim, num_classes)
        self.layer_norm_v       = nn.LayerNorm(vision_dim)
        self.layer_norm_t       = nn.LayerNorm(text_dim)

        freeze_weights = int(config["freeze-weights"])
        self.freeze_weights     = True if freeze_weights == 1 else False

        if self.freeze_weights:
            for p in self.vision_model.parameters():
                p.requires_grad = False

            for p in self.text_model.parameters():
                p.requires_grad = False

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.prediction_threshold = prediction_threshold
        self.val_auroc = MultilabelAUROC(
            num_labels=self.num_classes,
            average="macro",
            #thresholds=[self.prediction_threshold]
        )

        self.val_ap_macro = MultilabelAveragePrecision(
            num_labels=self.num_classes,
            average="macro",
            #thresholds=[self.prediction_threshold]
        )

        self.val_ap_micro = MultilabelAveragePrecision(
            num_labels=self.num_classes,
            average="micro",
            #thresholds=[self.prediction_threshold]
        )

        self.val_f1 = MultilabelF1Score(
            num_labels=self.num_classes,
            average="macro",
            #threshold=self.prediction_threshold
        )

        self.test_auroc = MultilabelAUROC(
            num_labels=self.num_classes,
            average="macro",
            #thresholds=[self.prediction_threshold]
        )

        self.test_ap = MultilabelAveragePrecision(
            num_labels=self.num_classes,
            average="macro",
            #thresholds=[self.prediction_threshold]
        )

        self.test_ap_micro = MultilabelAveragePrecision(
            num_labels=self.num_classes,
            average="micro",
            #thresholds=[self.prediction_threshold]
        )

        self.test_f1 = MultilabelF1Score(
            num_labels=self.num_classes,
            average="macro",
            #threshold=self.prediction_threshold
        )

        self.cm_test = MultilabelConfusionMatrix(num_labels=self.num_classes)

    def forward(self, x):
        pixel_values = x["pixel_values"]
        input_ids = x["input_ids"].to(self.device)
        attention_mask = x["attention_mask"].to(self.device)

        with torch.no_grad():
            img_embeds = self.vision_model(pixel_values)          
            outputs_t  = self.text_model(input_ids, attention_mask)

        txt_embeds = outputs_t.last_hidden_state[:, 0]            

        img_embeds = self.layer_norm_v(img_embeds)
        txt_embeds = self.layer_norm_t(txt_embeds)

        logits = self.fusion(img_embeds, txt_embeds)
        return logits
    
    def training_step(self, batch, batch_idx):
        images, metadata, labels = batch        
        x = {
            "pixel_values": images,
            "input_ids": metadata["input_ids"],
            "attention_mask": metadata["attention_mask"]
        }
        logits = self(x)
        loss = self.loss_fn(logits, labels.float())
        self.log("train_loss", loss, prog_bar=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        images, metadata, labels = batch
        x = {
            "pixel_values": images,
            "input_ids": metadata["input_ids"],
            "attention_mask": metadata["attention_mask"]
        }
        logits = self(x)
        loss = self.loss_fn(logits, labels.float())
        probs = torch.sigmoid(logits) # preds = probs here

        self.val_auroc.update(probs, labels.int())
        self.val_ap_macro.update(probs, labels.int())
        self.val_ap_micro.update(probs, labels.int())
        self.val_f1.update(probs, labels.int())

        self.log("val_loss", loss, prog_bar=True, batch_size=images.size(0))
        return loss
    
    def on_validation_epoch_end(self):
        self.log("val_auroc", self.val_auroc.compute(), prog_bar=True)
        self.log("val_map", self.val_ap_macro.compute(), prog_bar=True)
        self.log("val_map_micro", self.val_ap_micro.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)

        self.val_auroc.reset()
        self.val_ap_macro.reset()
        self.val_ap_micro.reset()
        self.val_f1.reset()
        
    def test_step(self, batch, batch_idx):
        images, metadata, labels = batch        
        x = {
            "pixel_values": images,
            "input_ids": metadata["input_ids"],
            "attention_mask": metadata["attention_mask"]
        }        
        logits = self(x)
        loss = self.loss_fn(logits, labels.float())
        probs = torch.sigmoid(logits)

        self.test_auroc.update(probs, labels.int())
        self.test_ap.update(probs, labels.int())
        self.test_ap_micro.update(probs, labels.int())
        self.test_f1.update(probs, labels.int())
        self.cm_test.update(probs, labels.int())
        self.log("test_loss", loss, batch_size=images.size(0))
        return loss
    
    def on_test_epoch_start(self):
        self.test_auroc.reset()
        self.test_ap.reset()
        self.test_ap_micro.reset()
        self.test_f1.reset()
    
    def on_test_epoch_end(self):
        self.log("test_auroc", self.test_auroc.compute())
        self.log("test_map", self.test_ap.compute())
        self.log("test_map_micro", self.test_ap_micro.compute())
        self.log("test_f1", self.test_f1.compute())    

        fig_, ax_ = self.cm_test.plot(
            labels=PATHOLOGY_COLUMNS
            )
        fig_.set_size_inches(24, 24)
        
        plt.close(fig_)
        plt.tight_layout()
        plt.savefig('Test-Resnetbert.png', dpi=300)
        plt.close(fig_)

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()), 
                lr=self.lr
                )
        elif self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()), 
                lr=self.lr
                )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), 
                lr=self.lr
                )
        return optimizer
    
    def predict(self, batch):
        images, metadata, labels = batch
        x = {
            "pixel_values": images,
            "input_ids": metadata["input_ids"],
            "attention_mask": metadata["attention_mask"]
        }        
        logits = self(x)
        probs = torch.sigmoid(logits)
        preds = (probs > self.prediction_threshold)
        return preds        
            
    def predict_step(self, x):
        images, metadata, labels = x        
        x = {
            "pixel_values": images,
            "input_ids": metadata["input_ids"],
            "attention_mask": metadata["attention_mask"]
        }        
        # change logits into probabilities and later take only over the threshold
        logits = self(x)
        probs = torch.sigmoid(logits)
        preds = (probs > self.prediction_threshold)
        return preds

    
class ResnetTabMultiModalMultilabelModel(L.LightningModule):
    def __init__(
        self,
        config,
        num_classes: int,
        prediction_threshold: float = 0.5,
        lr: float = 1e-3,
        optimizer: str = "AdamW"
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.lr = lr
        self.optimizer_name = optimizer

        self.vision_model_name = config["image-model"]
        self.tabular_model_config = config["tab-model"]

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        vision_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()   

        self.vision_model   = resnet
        self.tabular_model  = TabularTransformerModel(self.tabular_model_config)
        tab_dim             = self.tabular_model.embed_dim
        self.layer_norm_v   = nn.LayerNorm(vision_dim)
        self.layer_norm_t   = nn.LayerNorm(tab_dim)
        self.fusion         = FusionModule(vision_dim + tab_dim, num_classes)

        freeze_weights = int(config["freeze-weights"])
        self.freeze_weights = True if freeze_weights == 1 else False

        if self.freeze_weights:
            for p in self.vision_model.parameters():
                p.requires_grad = False
            
        # due to multilabel logic - this should be the loss
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.prediction_threshold = prediction_threshold
        self.val_auroc = MultilabelAUROC(
            num_labels=self.num_classes,
            average="macro",
            #thresholds=[self.prediction_threshold]
        )

        self.val_ap = MultilabelAveragePrecision(
            num_labels=self.num_classes,
            average="macro",
            #thresholds=[self.prediction_threshold]
        )

        self.val_ap_micro = MultilabelAveragePrecision(
            num_labels=self.num_classes,
            average="micro",
            #thresholds=[self.prediction_threshold]
        )


        self.val_f1 = MultilabelF1Score(
            num_labels=self.num_classes,
            average="macro",
            #threshold=self.prediction_threshold
        )

        self.test_auroc = MultilabelAUROC(
            num_labels=self.num_classes,
            average="macro",
            #thresholds=[self.prediction_threshold]
        )

        self.test_ap = MultilabelAveragePrecision(
            num_labels=self.num_classes,
            average="macro",
            #thresholds=[self.prediction_threshold]
        )

        self.test_ap_micro = MultilabelAveragePrecision(
            num_labels=self.num_classes,
            average="micro",
            #thresholds=[self.prediction_threshold]
        )

        self.test_f1 = MultilabelF1Score(
            num_labels=self.num_classes,
            average="macro",
            #threshold=self.prediction_threshold
        )

        self.cm_test = MultilabelConfusionMatrix(num_labels=self.num_classes)

    def forward(self, x):
        pixel_values = x["pixel_values"]
        tabular_values = x["tabular_values"]

        with torch.no_grad():
            outputs_v = self.vision_model(pixel_values)
        
        tab_embeds = self.tabular_model(tabular_values)

        tab_embeds = self.layer_norm_t(tab_embeds)
        img_embeds = self.layer_norm_v(outputs_v)

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
        probs = torch.sigmoid(logits)

        self.val_auroc.update(probs, labels.int())
        self.val_ap.update(probs, labels.int())
        self.val_ap_micro.update(probs, labels.int())
        self.val_f1.update(probs, labels.int())

        self.log("val_loss", loss, prog_bar=True, batch_size=images.size(0))
        return loss
    
    def on_validation_epoch_end(self):
        self.log("val_auroc", self.val_auroc.compute(), prog_bar=True)
        self.log("val_map", self.val_ap.compute(), prog_bar=True)
        self.log("val_map_micro", self.val_ap_micro.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)

        self.val_auroc.reset()
        self.val_ap.reset()
        self.val_ap_micro.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        images, metadata, labels = batch
        x = {
            "pixel_values": images,
            "tabular_values": metadata
        }
        logits = self(x)
        loss = self.loss_fn(logits, labels.float())
        probs = torch.sigmoid(logits)

        self.test_auroc.update(probs, labels.int())
        self.test_ap.update(probs, labels.int())
        self.test_ap_micro.update(probs, labels.int())
        self.test_f1.update(probs, labels.int())
        self.cm_test.update(probs, labels.int())
        self.log("test_loss", loss, batch_size=images.size(0))
        return loss
    
    def on_test_epoch_start(self):
        self.test_auroc.reset()
        self.test_ap.reset()
        self.test_ap_micro.reset()
        self.test_f1.reset()
    
    def on_test_epoch_end(self):
        self.log("test_auroc", self.test_auroc.compute())
        self.log("test_map", self.test_ap.compute())
        self.log("test_map_micro", self.test_ap_micro.compute())
        self.log("test_f1", self.test_f1.compute())
        fig_, ax_ = self.cm_test.plot(
            labels=PATHOLOGY_COLUMNS
            )
        fig_.set_size_inches(24, 24)
        plt.tight_layout()
        plt.savefig('Test-Resnettab.png', dpi=300)
        plt.close(fig_)
        
    def predict(self, batch):
        images, metadata, _ = batch
        x = {
            "pixel_values": images,
            "tabular_values": metadata
        }
        logits = self(x)
        probs = torch.sigmoid(logits)
        preds = (probs > self.prediction_threshold)
        return preds        
        
    def predict_step(self, x):
        images, metadata, labels = x
        x = {
            "pixel_values": images,
            "tabular_values": metadata
        }
        # change logits into probabilities and later take only over the threshold
        logits = self(x)
        probs = torch.sigmoid(logits)
        preds = (probs > self.prediction_threshold)
        return preds

    
    def configure_optimizers(self):
        if self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()), 
                lr=self.lr
                )
        if self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()), 
                lr=self.lr
                )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), 
                lr=self.lr
                )
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
    
class ResnetMonoMultilabelModel(L.LightningModule):
    def __init__(
        self,
        config,
        num_classes: int,
        prediction_threshold: float = 0.5,
        lr: float = 1e-3,
        optimizer: str = "AdamW"
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.lr = lr
        self.optimizer_name = optimizer

        self.vision_model_name = config["image-model"]

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        vision_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()           

        self.vision_model = resnet
        in_size = vision_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_size, in_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_size // 2, num_classes),
        )

        freeze_weights = int(config["freeze-weights"])
        self.freeze_weights = True if freeze_weights == 1 else False

        if self.freeze_weights:
            for p in self.vision_model.parameters():
                p.requires_grad = False

        # due to multilabel logic - this should be the loss
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.prediction_threshold = prediction_threshold
        self.val_auroc = MultilabelAUROC(
            num_labels=self.num_classes,
            average="macro",
            #thresholds=[self.prediction_threshold]
        )

        self.val_ap = MultilabelAveragePrecision(
            num_labels=self.num_classes,
            average="macro",
            #thresholds=[self.prediction_threshold]
        )

        self.val_ap_micro = MultilabelAveragePrecision(
            num_labels=self.num_classes,
            average="micro",
            #thresholds=[self.prediction_threshold]
        )

        self.val_f1 = MultilabelF1Score(
            num_labels=self.num_classes,
            average="macro",
            #threshold=self.prediction_threshold
        )

        self.test_auroc = MultilabelAUROC(
            num_labels=self.num_classes,
            average="macro",
            #thresholds=[self.prediction_threshold]
        )

        self.test_ap = MultilabelAveragePrecision(
            num_labels=self.num_classes,
            average="macro",
            #thresholds=[self.prediction_threshold]
        )

        self.test_ap_micro = MultilabelAveragePrecision(
            num_labels=self.num_classes,
            average="micro",
            #thresholds=[self.prediction_threshold]
        )

        self.test_f1 = MultilabelF1Score(
            num_labels=self.num_classes,
            average="macro",
            #threshold=self.prediction_threshold
        )

        self.cm_test = MultilabelConfusionMatrix(num_labels=self.num_classes)

    def forward(self, x):
        with torch.no_grad():
            outputs_v = self.vision_model(x["pixel_values"])
        
        logits = self.classifier(outputs_v)
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
    
    # def on_train_start(self):
    #     self.vision_model.train()        

    def validation_step(self, batch, batch_idx):
        images, _, labels = batch
        x = {
            "pixel_values": images
        }
        logits = self(x)
        loss = self.loss_fn(logits, labels.float())
        probs = torch.sigmoid(logits)

        self.val_auroc.update(probs, labels.int())
        self.val_ap.update(probs, labels.int())
        self.val_ap_micro.update(probs, labels.int())
        self.val_f1.update(probs, labels.int())

        self.log("val_loss", loss, prog_bar=True, batch_size=images.size(0))
        return loss
    
    def on_validation_epoch_end(self):
        self.log("val_auroc", self.val_auroc.compute(), prog_bar=True)
        self.log("val_map", self.val_ap.compute(), prog_bar=True)
        self.log("val_map_micro", self.val_ap_micro.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)

        self.val_auroc.reset()
        self.val_ap.reset()
        self.val_ap_micro.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        images, _, labels = batch
        x = {
            "pixel_values": images
        }
        logits = self(x)
        loss = self.loss_fn(logits, labels.float())
        probs = torch.sigmoid(logits)

        self.test_auroc.update(probs, labels.int())
        self.test_ap.update(probs, labels.int())
        self.test_ap_micro.update(probs, labels.int())
        self.test_f1.update(probs, labels.int())
        self.cm_test.update(probs, labels.int())
        self.log("test_loss", loss, batch_size=images.size(0))
        return loss
    
    def on_test_epoch_start(self):
        self.test_auroc.reset()
        self.test_ap.reset()
        self.test_ap_micro.reset()
        self.test_f1.reset()

    def on_test_epoch_end(self):
        self.log("test_auroc", self.test_auroc.compute())
        self.log("test_map", self.test_ap.compute())
        self.log("test_map_micro", self.test_ap_micro.compute())
        self.log("test_f1", self.test_f1.compute())
        fig_, ax_ = self.cm_test.plot(
            labels=PATHOLOGY_COLUMNS
            )
        fig_.set_size_inches(24, 24)
        plt.tight_layout()
        plt.savefig('Test-Resnet.png', dpi=300)
        plt.close(fig_)
        
    def predict(self, batch):
        images, _, _ = batch
        x = {
            "pixel_values": images
        }
        logits = self(x)
        probs = torch.sigmoid(logits)
        preds = (probs > self.prediction_threshold)
        return preds        
        
    def predict_step(self, x):
        images, _, _  = x
        x = {
            "pixel_values": images
        }
        # change logits into probabilities and later take only over the threshold
        logits = self(x)
        probs = torch.sigmoid(logits)
        preds = (probs > self.prediction_threshold)
        return preds

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer