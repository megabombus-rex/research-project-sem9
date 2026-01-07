# this should be a combination of a Vision Transformer (ViT) and a text embedding transformer with some sort of a fusion layer

from sklearn.metrics import precision_score, f1_score, recall_score
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModel, AutoTokenizer


def configure_optimizer(
    params, lr: float = 1e-3, optimizer: str = "Adam", momentum: float = 0.5
):
    if optimizer == "Adam":
        return torch.optim.Adam(params, lr=lr)
    if optimizer == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    raise KeyError(f"Optimizer '{optimizer}' is not supported.")


class FusionModule(nn.Module):
    def __init__(self, in_size, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_size, in_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_size // 2, num_classes),
        )

    def forward(self, img_embeds, txt_embeds):
        fused = torch.cat([img_embeds, txt_embeds], dim=-1)
        return self.fc(fused)


class ViTTextMultiModalModel(nn.Module):
    def __init__(
        self, config, num_classes: int, lr: float = 1e-3, optimizer: str = "Adam"
    ):
        super().__init__()
        self.num_classes = num_classes

        self.vision_model_name = config["image-model"]
        self.text_model_name = config["text-model"]
        self.tokenizer_name = config["text-model"]

        self.vision_model = AutoModel.from_pretrained(self.vision_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.text_model = AutoModel.from_pretrained(self.text_model_name)
        self.fusion = FusionModule(vision_dim + text_dim, num_classes)

        vision_dim = self.vision_model.config.hidden_size
        text_dim = self.text_model.config.hidden_size

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer = configure_optimizer(self.parameters(), lr, optimizer)

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

    # important
    def compute_loss(self, batch):
        logits = self(batch)
        labels = batch["labels"]
        return self.loss_fn(logits, labels)

    
class ViTTabMultiModalMutilabelModel(nn.Module):
    def __init__(self, config, num_classes: int, lr: float = 1e-3, optimizer: str = 'Adam'):
        super().__init__()
        self.num_classes = num_classes
        
        self.vision_model_name = config['image-model']
        self.tabular_model_config = config['tab-model']        
        
        
        self.vision_model   = AutoModel.from_pretrained(self.vision_model_name)
        self.tabular_model  = TabularTransformerModel(self.tabular_model_config)
        vision_dim, tab_dim = (self.vision_model.config.hidden_size, self.tabular_model.embed_dim)
        self.fusion         = FusionModule(vision_dim + tab_dim, num_classes)
        
    def forward(self, x):
        pixel_values = x['pixel_values']
        table_values = x['table_values']
        
        outputs_v = self.vision_model(pixel_values)
        outputs_t = self.tabular_model(table_values)
        img_embeds = outputs_v.last_hidden_state[:, 0]
        txt_embeds = outputs_t.last_hidden_state[:, 0]
        
        logits = self.fusion(img_embeds, txt_embeds)
        return logits
        
        
        
class TabularTransformerModel(nn.Module):    
    def __init__(self, model_config):
        super().__init__()
        embed_dim = model_config.get("embed_dim", 64)
        depth = model_config.get("depth", 3)
        num_heads = model_config.get("num_heads", 4)
        dropout = model_config.get("dropout", 0.1)
        self.lr = model_config.get("learning_rate", 1e-4)
        
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
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = self.embed(x.unsqueeze(-1))  # (batch, features, embed_dim)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        embeds = self.transformer(x)
        return embeds 