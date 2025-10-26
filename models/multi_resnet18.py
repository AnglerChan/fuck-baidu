import torch
import torch.nn as nn
import torchvision.models as models

class MultiModalResNet18(nn.Module):
    def __init__(self, num_classes=13, pretrained_path='models/resnet18-f37072fd.pth'):
        super().__init__()
        # 三个独立的resnet18
        self.backbone_color = models.resnet18(weights=None)
        self.backbone_depth = models.resnet18(weights=None)
        self.backbone_infrared = models.resnet18(weights=None)

        print(f"Loading pretrained weights from: {pretrained_path}")
        state_dict = torch.load(pretrained_path)
        self.backbone_color.load_state_dict(state_dict)

        # 去掉最后分类层，保留特征
        self.backbone_color.fc = nn.Identity()
        self.backbone_depth.fc = nn.Identity()
        self.backbone_infrared.fc = nn.Identity()

        # # 拼接后的特征是 512 * 3 = 1536
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 3, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, num_classes)
        # )
        self.classifier = nn.Linear(512*3, num_classes)

    def forward(self, x_color, x_depth, x_ir):
        f_color = self.backbone_color(x_color)
        f_depth = self.backbone_depth(x_depth)
        f_ir = self.backbone_infrared(x_ir)

        f_cat = torch.cat([f_color, f_depth, f_ir], dim=1)
        out = self.classifier(f_cat)
        return out
