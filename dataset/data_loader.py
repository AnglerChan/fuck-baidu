import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class MultiModalDataset(Dataset):
    """
    - 训练集：有 label 文件
    - 测试集：无 label 文件
    """
    def __init__(self, root, label_file=None, transform=None, is_train=True):
        self.root = root
        self.transform = transform
        self.modalities = ["color", "depth", "infrared"]
        self.is_train = is_train

        color_dir = os.path.join(root, "color")
        self.samples = sorted([
            f for f in os.listdir(color_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        # 如果有标签文件，则读取
        self.labels = {}
        if label_file is not None and os.path.exists(os.path.join(root, label_file)):
            with open(os.path.join(root, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        fname, cls = parts
                        self.labels[fname] = int(cls)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname = self.samples[idx]
        imgs = []
        for m in self.modalities:
            img_path = os.path.join(self.root, m, fname)
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        if self.is_train:
            label = self.labels.get(fname)
            if label is None:
                raise ValueError(f"Missing label for {fname}")
            return imgs[0], imgs[1], imgs[2], label
        else:
            return imgs[0], imgs[1], imgs[2], fname  # 推理模式返回文件名
