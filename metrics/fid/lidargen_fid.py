import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
from .fid_score import calculate_frechet_distance


def reduction_strategy(array, indices):

    #array = array[0]
    #array = np.transpose(array, (2, 0, 1))
    #array = array.reshape((1024, 32*64))
    #array = np.mean(array, axis=1)

    # return array

    # Random Reduction
    return array.reshape((-1))[indices]


def load_activations(load, indices):
    all_files = glob.glob(load + "/*")
    all_arrays = []

    for file in all_files:
        x = np.load(file)  # torch.load(file).numpy()
        x = reduction_strategy(x, indices)
        all_arrays.append(x)

    all_activations = np.stack(all_arrays, axis=0)

    return all_activations[0:1100]


def compute_stats(load, indices):
    all_activations = load_activations(load, indices)

    mu = np.mean(all_activations, axis=0)
    sigma = np.cov(all_activations, rowvar=False)

    return mu, sigma


def get_fid(folder1, folder2):

    random.seed(0)
    indices = random.sample(range(0, 2097152), 4096)  # 2048)

    mu1, sigma1 = compute_stats(folder1, indices)
    mu2, sigma2 = compute_stats(folder2, indices)

    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    return fid_value
