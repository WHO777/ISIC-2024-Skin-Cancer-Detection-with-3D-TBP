import timm
import torch


class TimmModel(torch.nn.Module):
    def __init__(self, name: str, pretrained: bool, **kwargs):
        super(TimmModel, self).__init__()
        self.model = timm.create_model(name, pretrained=pretrained, **kwargs)

    def forward(self, x):
        return self.model(x, return_dict=False)


class TimmBackbone(TimmModel):
    def __init__(self, name: str, pretrained: bool, **kwargs):
        super().__init__(name, pretrained, **kwargs)
        self.model.global_pool = torch.nn.Identity()
        self.model.classifier = torch.nn.Identity()
        self.model.head = torch.nn.Identity()

    def forward(self, x):
        x = self.model.forward_features(x)
        return x

    @property
    def num_features(self) -> int:
        return self.model.num_features
