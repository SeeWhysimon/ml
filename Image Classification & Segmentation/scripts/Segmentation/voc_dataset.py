import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def get_train_loader(batch_size=16):
    input_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    target_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_data = datasets.VOCSegmentation(
        root='../../../data/Image Classification & Segmentation/VOC2012_train_val/',
        year='2012',
        image_set='train',
        download=True,
        transform=input_transform,
        target_transform=target_transform
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader