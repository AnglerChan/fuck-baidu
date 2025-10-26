from dataset.data_loader import MultiModalDataset
from torch.utils.data import DataLoader
from torchvision import transforms

def make_data_loader(args):
    # 数据增强
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    train_labels = args.labels
    val_labels = None  # args.val_labels
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 数据加载
    train_dataset = MultiModalDataset(
        root=args.root,
        label_file=train_labels,
        transform=transform_train,
        is_train=True
    )
    val_dataset = MultiModalDataset(
        root=args.root,
        label_file=val_labels,
        transform=transform_val,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_val, shuffle=False, num_workers=args.workers)



    return train_loader, val_loader