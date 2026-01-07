import torch
import torch.nn as nn

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