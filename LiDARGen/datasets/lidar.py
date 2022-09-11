from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import os
from glob import glob
from .lidar_utils import point_cloud_to_range_image


class LiDAR(Dataset):

    def __init__(self, path, config, resolution=None, transform=None):
        self.length = 1000
        self.transform = transform

        filename_real = os.path.join(path, '0_200.npy')

        #gen = np.load(filename_gen)
        real = np.load(filename_real)
        # Make negatives 0
        real = np.where(real < 0, 0, real) + 0.0001

        # Apply log
        real = ((np.log2(real+1)) / 6)

        # Make negatives 0
        real = np.clip(real, 0, 1)

        # output = torch.from_numpy(real).cuda()
        # zero mean
        real -= real.mean()
        # std dev 1
        real /= real.std()

        self.data = real.reshape(
            (1, config.data.image_size, config.data.image_width))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.data
        return data, 0
