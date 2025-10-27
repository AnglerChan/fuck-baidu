import torch
import torch.nn as nn
import torchvision.models as models

class MultiModalResNet18(nn.Module):
    def __init__(self, num_classes=13, pretrained_path='/data_C/minzhi/Projects/DaBang/models/resnet18-f37072fd.pth'):
        super().__init__()
        # 共享的resnet18
        self.backbone = models.resnet18(weights=None)

        print(f"Loading pretrained weights from: {pretrained_path}")  # yuxunlian
        state_dict = torch.load(pretrained_path)
        self.backbone.load_state_dict(state_dict)

        # 去掉最后分类层，保留特征
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Linear(512*3, num_classes)

    def forward(self, x_color, x_depth, x_ir):
        f_color = self.backbone(x_color)
        f_depth = self.backbone(x_depth)
        f_ir = self.backbone(x_ir)

        f_cat = torch.cat([f_color, f_depth, f_ir], dim=1)
        out = self.classifier(f_cat)
        return out
