
import pdb
import cv2
import numpy as np
import torch
import glob


def calculate_mae(exp_dir):
    error_bc = 0.0
    error_nn = 0.0
    error_ours = 0.0

    sample_count = len(glob.glob(exp_dir + '/densification_target/*.pth'))

    for idx in range(sample_count):
        result_path = (exp_dir + '/densification_result/{0}.pth'.format(str(idx)))
        target_path = (exp_dir + '/densification_target/{0}.pth'.format(str(idx)))

        result = torch.load(result_path).cpu().numpy()[0]
        target = torch.load(target_path).cpu().numpy()[0]

        result = np.exp2(result*6)-1
        target = np.exp2(target*6)-1

        result_bc = target[::4]
        result_bc = cv2.resize(result_bc, (0, 0), fx=1.0, fy=4.0, interpolation=cv2.INTER_CUBIC)

        result_nn = target[::4]
        result_nn = cv2.resize(result_nn, (0, 0), fx=1.0, fy=4.0, interpolation=cv2.INTER_NEAREST)

        error_bc += np.sum(np.abs(result_bc - target))
        error_nn += np.sum(np.abs(result_nn - target))
        error_ours += np.sum(np.abs(result - target))

    count = sample_count * 1024 * 64

    error_bc = error_bc / count
    error_nn = error_nn / count
    error_ours = error_ours / count

    print('-------------------------------------------------')
    print('MAE Bicubic:  ' + str(error_bc))
    print('MAE NN:       ' + str(error_nn))
    print('MAE LiDARGen: ' + str(error_ours))
    print('-------------------------------------------------')
