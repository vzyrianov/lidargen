import torch
import numpy as np
import glob
import pdb

from sklearn.metrics import jaccard_score

def calculate_iou(model_dir, expected_dir):
   
   file_count = len(glob.glob(expected_dir + '/*'))

   seg_list_gt = []
   for i in range(0, file_count):
   	seg_list_gt.append(torch.load(expected_dir + '/' + str(i) + '.pth'))
   
   seg_gt = torch.stack(seg_list_gt)
   seg_gt = seg_gt.flatten().cpu().detach().numpy()
   
   seg_list_nn = []
   for i in range(0, file_count):
   	seg_list_nn.append(torch.load(model_dir + '/' + str(i) + '.pth'))
   
   seg_nn = torch.stack(seg_list_nn)
   seg_nn = seg_nn.flatten().cpu().detach().numpy()
   

   return jaccard_score(seg_gt, seg_nn, average='weighted')



