# src/dataset.py

import torch

from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

from config import IMAGE_SIZE, BATCH_SIZE


def get_train_transforms(image_size=IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transforms(image_size=IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def create_dataloaders(data_dir):
    train_full = datasets.ImageFolder(data_dir, transform=get_train_transforms())
    val_full = datasets.ImageFolder(data_dir, transform=get_val_transforms())
    test_full = datasets.ImageFolder(data_dir, transform=get_val_transforms())

    class_names = train_full.classes

    train_size = int(0.7 * len(train_full))
    val_size = int(0.15 * len(train_full))
    test_size = len(train_full) - train_size - val_size

    generator = None

    train_subset, val_subset, test_subset = random_split(
        range(len(train_full)),
        [train_size, val_size, test_size],
        generator=generator
    )

    train_dataset = torch.utils.data.Subset(train_full, train_subset.indices)
    val_dataset = torch.utils.data.Subset(val_full, val_subset.indices)
    test_dataset = torch.utils.data.Subset(test_full, test_subset.indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, class_names