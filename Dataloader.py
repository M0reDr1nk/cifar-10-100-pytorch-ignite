from torch.utils.data import DataLoader
import torchvision.transforms as Transforms
from torchvision.datasets import CIFAR10,CIFAR100
import torch
def get_data_loaders(train_batch_size, val_batch_size):
    data_transform_train = Transforms.Compose([

        Transforms.RandomCrop(32, padding=4),
        Transforms.RandomHorizontalFlip(0.5),
        Transforms.RandomVerticalFlip(0.5),
        Transforms.RandomRotation(15,expand=True),
        Transforms.CenterCrop(32),
        #Transforms.Resize(48),
        Transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        Transforms.ToTensor(),
        Transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    )
    data_transform_test = Transforms.Compose([
        Transforms.Resize(32),
        Transforms.ToTensor(),
        Transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    )

    train_data = CIFAR10(root="./dataset/", transform=data_transform_train, target_transform=None,train=True)
    train_loader=torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=4,pin_memory=True)

    val_data = CIFAR10(root="./dataset/", transform=data_transform_test, target_transform=None,train=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, shuffle=False, num_workers=4,
                                               pin_memory=True)
    return train_loader, val_loader
