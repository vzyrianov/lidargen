from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import os
from glob import glob
from .lidar_utils import point_cloud_to_range_image

class KITTI(Dataset):

    def __init__(self, path, config, split = 'train', resolution=None, transform=None):
        self.transform = transform
        self.return_remission = (config.data.channels == 2)
        self.random_roll = config.data.random_roll
        full_list = glob(os.path.join(os.environ.get('KITTI360_DATASET'), 'data_3d_raw/*/velodyne_points/data/*.bin'))
        if split == "train":
            self.full_list = list(filter(lambda file: '0000_sync' not in file and '0001_sync' not in file, full_list))
        else:
            self.full_list = list(filter(lambda file: '0000_sync' in file or '0001_sync' in file, full_list))
        self.length = len(self.full_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        filename = self.full_list[idx]
        if self.return_remission:
            real, intensity = point_cloud_to_range_image(filename, False, self.return_remission)
        else:
            real = point_cloud_to_range_image(filename, False, self.return_remission)
        #Make negatives 0
        real = np.where(real<0, 0, real) + 0.0001
        #Apply log
        real = ((np.log2(real+1)) / 6)
        #Make negatives 0
        real = np.clip(real, 0, 1)
        random_roll = np.random.randint(1024)

        if self.random_roll:
            real = np.roll(real, random_roll, axis = 1)
        real = np.expand_dims(real, axis = 0)

        if self.return_remission:
            intensity = np.clip(intensity, 0, 1.0)
            if self.random_roll:
                intensity = np.roll(intensity, random_roll, axis = 1)
            intensity = np.expand_dims(intensity, axis = 0)
            real = np.concatenate((real, intensity), axis = 0)

        return real, 0

