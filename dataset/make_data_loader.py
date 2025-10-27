from dataset.data_loader import MultiModalDataset
from torch.utils.data import DataLoader
from torchvision import transforms

def make_data_loader(args):
    # 数据增强
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_labels = args.train_labels
    val_labels = args.val_labels

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
    )
    val_dataset = MultiModalDataset(
        root=args.root,
        label_file=val_labels,
        transform=transform_val,
    )
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True,
                              num_workers=args.workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch,
                            shuffle=False,
                            num_workers=args.workers)



    return train_loader, val_loader

def make_test_data_loader(args):
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = MultiModalDataset(
        root=args.root,
        transform=transform_test,
        is_train=False
    )

    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch,
                            shuffle=False,
                            num_workers=args.workers)
    return test_loader