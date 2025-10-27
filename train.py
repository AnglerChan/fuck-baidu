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
from utils.meter import AverageMeter
from utils.utils import setup_logger

from models.multi_resnet18 import MultiModalResNet18
os.environ['CUDA_VISIBLE_DEVICES'] = '5'


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
    set_seed(2025)
    output_file = args.output_file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("mmc", output_file, if_train=True)
    if output_file and not os.path.exists(output_file):
        os.makedirs(output_file)

    # data
    train_loader, val_loader = make_data_loader(args)
    # 模型
    model = MultiModalResNet18(num_classes=args.num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0

    print(f"Logging to {output_file}")
    logger.info(f"Dataset: {args.root}")
    logger.info(f"Num epochs: {args.epochs}, Batch size: {args.batch}, LR: {args.lr}")
    logger.info(f"Device: {device}\n")
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"\n\n========== Training Started: {start_time} ==========\n")

    loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    val_acc_meter = AverageMeter()

    for epoch in range(args.epochs):
        loss_meter.reset()
        val_loss_meter.reset()
        acc_meter.reset()
        val_acc_meter.reset()

        model.train()
        # running_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for x_c, x_d, x_ir, labels in pbar:
            x_c, x_d, x_ir, labels = x_c.to(device), x_d.to(device), x_ir.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(x_c, x_d, x_ir)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            loss_meter.update(loss.item(), x_c.shape[0])
            acc = (outputs.max(1)[1] == labels).float().mean()
            acc_meter.update(acc, 1)
            # running_loss += loss.item()
            # _, preds = torch.max(outputs, 1)
            # correct += (preds == labels).sum().item()
            # total += labels.size(0)
            # acc = correct / total

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc*100:.2f}%")

        avg_loss = loss_meter.avg  # running_loss / len(train_loader)
        avg_acc = acc_meter.avg  # correct / total
        logger.info( f"Epoch [{epoch+1}/{args.epochs}]"
                     f" Train Loss: {avg_loss:.4f} Train Acc: {avg_acc*100:.2f}%")
        if epoch % args.eval_period == 0:
            model.eval()
            with torch.no_grad():
                for x_c, x_d, x_ir, labels in val_loader:
                    x_c, x_d, x_ir, labels = x_c.to(device), x_d.to(device), x_ir.to(device), labels.to(device)
                    outputs = model(x_c, x_d, x_ir)
                    val_loss = criterion(outputs, labels)
                    val_loss_meter.update(val_loss.item(), x_c.shape[0])

                    val_acc = (outputs.max(1)[1] == labels).float().mean()
                    val_acc_meter.update(val_acc, 1)
        logger.info(f"Epoch [{epoch + 1}/{args.epochs}] "
              f"Val Loss {val_loss_meter.avg:.4f} Val Acc {val_acc_meter.avg:.4f}")

        # 保存最优模型
        if val_acc_meter.avg > best_acc:
            best_acc = val_acc_meter.avg
            save_path = os.path.join(args.output_file, "best_model.pth")
            torch.save({'state_dict': model.state_dict()}, save_path)
            logger.info( f"New best model saved ({best_acc*100:.2f}%)")

    logger.info(f"\ntraining finished. Best val Acc: {best_acc*100:.2f}%")
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info( f"\n========== Training Ended: {end_time} ==========\n")


# -------------------------------
#  参数定义
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multi-modal ResNet18 classifier")
    parser.add_argument('--root', type=str, default='/data_C/minzhi/datasets/MMOC/train_2k', help='Training dataset root path (contains color/depth/infrared)')
    parser.add_argument('--train_labels', type=str, default='new_train_labels.txt', help='Label file name')
    parser.add_argument('--val_labels', type=str, default='val_labels.txt', help='Label file name')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
    parser.add_argument('--eval_period', type=int, default=1, help='val per 1 epochs')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=13, help='Number of classes')
    parser.add_argument('--output_file', type=str, default='/data_C/minzhi/Projects/DaBang/logs/4', help='Save checkpoint path')
    parser.add_argument('--workers', type=int, default=8, help='Number of DataLoader workers')
    parser.add_argument('--no_pretrain', action='store_true', help='Disable pretrained weights')
    args = parser.parse_args()

    train(args)
