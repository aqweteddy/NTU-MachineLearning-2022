import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
import random, os
from PIL import Image
import torch
from torch.nn import functional as F
from typing import List


def get_all_paths(path):
    files = sorted([
        os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")
    ])
    return files


def get_dataloader(train_paths, val_paths, train_tfm, test_tfm, batch_size):
    train_ds = FoodDataset(train_paths, tfm=train_tfm)
    val_ds = FoodDataset(val_paths, tfm=test_tfm)
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True)
    valid_loader = DataLoader(val_ds,
                              batch_size=batch_size * 2,
                              shuffle=False,
                              num_workers=8,
                              pin_memory=True)

    return train_loader, valid_loader


def get_transform(mode='rand'):
    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.AutoAugment(
            transforms.autoaugment.AutoAugmentPolicy.IMAGENET)
        if mode == 'auto' else transforms.RandAugment(
            num_ops=3, magnitude=9, num_magnitude_bins=31),
        # You may add some transforms here.
        # ToTensor() should be the last one of the transforms.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tfm, test_tfm


class FoodDataset(Dataset):
    num_classes = 11

    def __init__(self, files: List[str], tfm):
        super(FoodDataset).__init__()
        self.files = files
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def get_raw(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        try:
            label = int(fname.split("/")[-1].split("_")[0])
            label = F.one_hot(torch.tensor(label),
                              self.num_classes).type(torch.float32)
        except:
            label = None  # test has no label

        return im, label

    def __getitem__(self, idx):
        im, l = self.get_raw(idx)
        if l is None:
            l = -1
        if self.transform is not None:
            im = self.transform(im)
        return im, l