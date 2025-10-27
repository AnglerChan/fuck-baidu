import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict


class MultiModalDataset(Dataset):
    def __init__(
        self,
        root: str,  # 对应 MMOC\train_2k 或者 test_1k
        label_file: Optional[str] = None,  # 标签文件名（train_labels.txt 或 val_labels.txt）
        transform=None,
        modalities: List[str] = ["color", "depth", "infrared"],  # 固定三模态
        is_train: bool = True  # test 预测的时候才false
    ):
        self.root = root
        self.transform = transform
        self.modalities = modalities
        self.is_train = is_train

        # 检查模态文件夹是否存在
        self._check_modality_dirs()

        # 加载所有样本（以color模态为基准，确保三模态文件匹配）
        self.samples = self._load_samples()

        # 加载标签（train/val模式），测试集无标签
        self.labels: Dict[str, int] = {}
        if self.is_train and label_file:
            self.labels = self._load_labels(label_file)
            # 过滤无标签的样本
            self.samples = [f for f in self.samples if f in self.labels]
            if not self.samples:
                raise ValueError(f"无匹配标签的样本，请检查 {label_file}")

    def _check_modality_dirs(self) -> None:
        for m in self.modalities:
            mod_dir = os.path.join(self.root, m)
            if not os.path.isdir(mod_dir):
                raise FileNotFoundError(f"模态文件夹不存在：{mod_dir}")

    def _load_samples(self) -> List[str]:
        # 以color模态为基准（可改为其他模态，确保文件唯一）
        base_mod = "color"
        base_dir = os.path.join(self.root, base_mod)
        all_files = [f for f in os.listdir(base_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not all_files:
            raise ValueError(f"color模态文件夹为空：{base_dir}")

        # 过滤在depth/infrared中缺失的文件
        valid_files = []
        for f in all_files:
            if all(os.path.exists(os.path.join(self.root, m, f)) for m in self.modalities):
                valid_files.append(f)
            else:
                print(f"文件 {f} 在部分模态中缺失")  # 正常不会出现
        return sorted(valid_files)

    def _load_labels(self, label_file: str) -> Dict[str, int]:
        label_path = os.path.join(self.root, label_file)  # 标签文件在root目录下
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"标签文件不存在：{label_path}")

        labels = {}
        with open(label_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    print(f"第{line_num}行格式错误，跳过：{line}")
                    continue
                fname, cls_str = parts
                try:
                    labels[fname.strip()] = int(cls_str)  # 清除空白字符
                except ValueError:
                    print(f"第{line_num}行标签无效，跳过：{cls_str}")
        return labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        fname = self.samples[idx]
        imgs = []
        for m in self.modalities:
            img_path = os.path.join(self.root, m, fname)
            img = Image.open(img_path).convert('RGB')  # 统一转为RGB（避免灰度图通道问题）
            if self.transform:
                img = self.transform(img)
            imgs.append(img)  # imgs顺序：[color, depth, infrared]

        if self.is_train:
            return (*imgs, self.labels[fname])  # 三模态图像 + 标签
        else:
            return (*imgs, fname)  # 测试模式返回文件名
