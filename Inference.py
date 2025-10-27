import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse

from dataset.make_data_loader import make_data_loader, make_test_data_loader
from models.multi_resnet18 import MultiModalResNet18
from dataset.data_loader import MultiModalDataset
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = make_test_data_loader(args)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = MultiModalResNet18(num_classes=args.num_classes)
    state_dict = torch.load(args.checkpoint, map_location=device)['state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for x_c, x_d, x_ir, fnames in tqdm(test_loader, desc="Predicting"):
            x_c, x_d, x_ir = x_c.to(device), x_d.to(device), x_ir.to(device)
            out = model(x_c, x_d, x_ir)
            preds = out.argmax(dim=1).cpu().numpy()
            for f, p in zip(fnames, preds):
                results.append((f, p))

    df = pd.DataFrame(results, columns=['filename', 'label_pred'])
    df.to_csv(args.save, index=False)
    print(f"Predictions saved to {args.save}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for multi-modal classifier")
    parser.add_argument('--root', type=str, default='/data_C/minzhi/datasets/MMOC/test_1k', help='Path to test dataset (color/depth/infrared folders)')
    parser.add_argument('--checkpoint', type=str, default='/data_C/minzhi/Projects/DaBang/logs/4/best_model.pth', help='Path to model checkpoint (.pth)')
    parser.add_argument('--save', type=str, default='logs/4/submission.csv', help='Output CSV path')
    parser.add_argument('--num_classes', type=int, default=13, help='Number of classes')
    parser.add_argument('--batch', type=int, default=32, help='batch_size')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')


    args = parser.parse_args()

    inference(args)
