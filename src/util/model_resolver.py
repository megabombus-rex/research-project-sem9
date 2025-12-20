from torch import nn
from torchvision.models import resnet50, resnet18, vgg16

def get_model_from_name(model_name: str, pretrained: bool) -> nn.Module | None:
    model_mapping = {
        "resnet-50": resnet50(pretrained=pretrained),
        "resnet-18": resnet18(pretrained=pretrained),
        "vgg-16": vgg16(pretrained=pretrained),
    }
    return model_mapping.get(model_name, None)