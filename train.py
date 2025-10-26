import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import argparse
import datetime

from dataset.make_data_loader import make_data_loader
from utils import setup_logger

from models.multi_resnet18 import MultiModalResNet18
os.environ['CUDA_VISIBLE_DEVICES'] = '6'


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
#  训练
def train(args):
    output_file = args.output_file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("mmc", output_file, if_train=True)
    set_seed(2025)

    # data
    train_loader, val_loader = make_data_loader(args)
    # 模型
    model = MultiModalResNet18(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    best_acc = 0

    logger.info(f"Dataset: {args.root}")
    logger.info(f"Num epochs: {args.epochs}, Batch size: {args.batch}, LR: {args.lr}")
    logger.info(f"Device: {device}\n")
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"\n\n========== Training Started: {start_time} ==========\n")
    print(f"Logging to {output_file}")

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for x_c, x_d, x_ir, labels in pbar:
            x_c, x_d, x_ir, labels = x_c.to(device), x_d.to(device), x_ir.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(x_c, x_d, x_ir)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            acc = correct / total

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc*100:.2f}%")

        avg_loss = running_loss / len(train_loader)
        avg_acc = correct / total
        logger.info( f"Epoch [{epoch+1}/{args.epochs}]  Loss: {avg_loss:.4f}  Acc: {avg_acc*100:.2f}%")

        # 保存最优模型
        if avg_acc > best_acc:
            best_acc = avg_acc
            save_path = os.path.join(args.output_file, "best_model.pth")
            torch.save({'state_dict': model.state_dict()}, save_path)
            logger.info( f"New best model saved ({best_acc*100:.2f}%)")

    logger.info(f"\nraining finished. Best training Acc: {best_acc*100:.2f}%")
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info( f"========== Training Ended: {end_time} ==========\n")


# -------------------------------
#  参数定义
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multi-modal ResNet18 classifier")
    parser.add_argument('--root', type=str, default='/data_C/minzhi/datasets/MM_Classification/train_2k', help='Training dataset root path (contains color/depth/infrared)')
    parser.add_argument('--labels', type=str, default='train_labels.txt', help='Label file name')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=13, help='Number of classes')
    parser.add_argument('--output_file', type=str, default='logs/2', help='Save checkpoint path')
    parser.add_argument('--workers', type=int, default=8, help='Number of DataLoader workers')
    parser.add_argument('--no_pretrain', action='store_true', help='Disable pretrained weights')
    args = parser.parse_args()

    train(args)
