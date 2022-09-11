from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import os
from glob import glob

class KITTI360(Dataset):

    def __init__(self, path, config, split = 'train', resolution=None, transform=None):
        self.transform = transform
        full_list = glob(os.path.join(path, 'data_2d_raw/*/image_00/data_rect/*.png'))
        if split == "train":
            self.full_list = list(filter(lambda file: '0000_sync' in file or '0001_sync' in file, full_list))
        else:
            self.full_list = list(filter(lambda file: '0000_sync' not in file, full_list))
        self.length = len(self.full_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        filename = self.full_list[idx]
        img = Image.open(filename)
        img = self.transform(img)
        return img, 0