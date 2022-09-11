import os
import torch
import torchvision.transforms as transforms
from datasets.kitti360_im import KITTI360
from datasets.lidar import LiDAR
from datasets.kitti import KITTI
from torch.utils.data import Subset
import numpy as np

def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])
    else:
        tran_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])


    if config.data.dataset == "lidar":
        if config.data.random_flip:
            dataset = LiDAR(path=os.path.join(args.exp, 'datasets', 'lidar'), config = config, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ]))
        else:
            dataset = LiDAR(path=os.path.join(args.exp, 'datasets', 'lidar'), config = config, transform=transforms.ToTensor())

        test_dataset = LiDAR(path=os.path.join(args.exp, 'datasets', 'lidar'), config = config, transform=transforms.ToTensor())

    elif config.data.dataset == "KITTI":
        if config.data.random_flip:
            dataset = KITTI(path=os.path.join(args.exp, 'datasets', 'KITTI'), config = config, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ]))
        else:
            dataset = KITTI(path=os.path.join(args.exp, 'datasets', 'KITTI'), config = config, transform=transforms.ToTensor())

        test_dataset = KITTI(path=os.path.join(args.exp, 'datasets', 'KITTI'), split = "test", config = config, transform=transforms.ToTensor())

    elif config.data.dataset == "KITTI360":
        dataset = KITTI360(path=os.path.join(args.exp, 'datasets', 'KITTI'), config = config, split = "train",
                            transform=transforms.Compose([
                                transforms.Resize((config.data.image_size, config.data.image_width)),
                                transforms.CenterCrop((config.data.image_size, config.data.image_width)),
                                transforms.ToTensor(),
                            ]))
        test_dataset = KITTI360(path=os.path.join(args.exp, 'datasets', 'KITTI'), config = config, split = "train",
                            transform=transforms.Compose([
                                transforms.Resize((config.data.image_size, config.data.image_width)),
                                transforms.CenterCrop((config.data.image_size, config.data.image_width)),
                                transforms.ToTensor(),
                            ]))

    return dataset, test_dataset

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)

def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]

    return X

def inverse_data_transform(config, X):
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.) / 2.

    return torch.clamp(X, 0.0, 1.0)
