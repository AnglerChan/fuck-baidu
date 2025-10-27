# utils.py
import torch
import os
from sklearn.metrics import classification_report
import json

import logging
import os
import sys
import os.path as osp


def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if if_train:
            fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='w')
        else:
            fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def save_checkpoint(state, is_best, checkpoint_dir='checkpoints', filename='checkpoint.pth'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_path)

def accuracy(output, target, topk=(1,)):
    """ compute top-k accuracy """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k.mul_(100.0 / batch_size)).item())
    return res

def evaluate_model(model, dataloader, device, num_classes=13):
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, pred = outputs.max(1)
            preds.extend(pred.cpu().numpy().tolist())
            gts.extend(labels.cpu().numpy().tolist())
    print(classification_report(gts, preds, digits=4))
    return preds, gts

def save_labels_csv(samples, out_path):
    """samples: list of (fname, label)"""
    import csv
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename','label'])
        for fname, lbl in samples:
            writer.writerow([fname,lbl])
