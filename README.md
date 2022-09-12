# Learning to Generate Realistic LiDAR Point Clouds (LiDARGen) [Paper](https://arxiv.org/abs/2204.13696)

This repository contains the official implementation of our paper [Learning to Generate Realistic LiDAR Point Clouds](http://www.zyrianov.org/lidargen). 

## Usage

### Environment
Install all python packages for training and evaluation with conda environment setup file: 
```bash
conda env create -f environment.yml
conda activate lidar3
```

### Set up KITTI-360

1. Download KITTI-360 from [http://www.cvlibs.net/datasets/kitti-360/](http://www.cvlibs.net/datasets/kitti-360/) (only the 3D LiDAR readings are required)
1. Set the KITTI360\_DATASET environment variable to the KITTI360 dataset path. `export KITTI360_DATASET=/path/to/dataset/KITTI-360/`.

### Sampling from Pretrained KITTI-360 model
1. Clone this repository and navigate to the project root directory
1. Download the pretrained model: `curl -L https://uofi.box.com/shared/static/ahnc453qpx6pa8o7ikt2rllr3cavwcob --output kitti_pretrained.tar.gz`
1. Extract the model: `tar -xzvf kitti_pretrained.tar.gz`
1. Setup conda environment: `conda env create --name lidargen --file=environment.yml`
1. Sample with `python lidargen.py --sample --exp kitti_pretrained --config kitti.yml`


## Metrics

### FID
Running FID requires downloading the 1024 backbone file for rangenet++ the following [link](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet53-1024.tar.gz)  and extracting it to the folder rangenetpp/lidar\_bonnetal\_master/darknet53-1024. This model is provided by the [RangeNet++ repository](https://github.com/PRBonn/lidar-bonnetal). Finally, lidargen needs to be run with the --fid option. This can be done by running the following commands:  

1. `curl -L http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet53-1024.tar.gz --output darknet53-1024.tar.gz`
1. `tar -xzvf darknet53-1024.tar.gz`
1. `mv darknet53-1024/backbone rangenetpp/lidar_bonnetal_master/darknet53-1024/`
1. `rm darknet53-1024.tar.gz`
1. `rm -r darknet53-1024`
1. `python lidargen.py --fid --exp kitti_pretrained --config kitti.yml`

## MMD

1. `python lidargen.py --mmd --exp kitti_pretrained --config kitti.yml`

## JSD

1. `python lidargen.py --jsd --exp kitti_pretrained --config kitti.yml`

## Denoising Visualization
To create a denoising visualization, LiDARGen needs to be run manually.

1. `cd LiDARGen`
1. `curl -L https://uofi.box.com/shared/static/ahnc453qpx6pa8o7ikt2rllr3cavwcob --output kitti_pretrained.tar.gz`
1. `tar -xzvf kitti_pretrained.tar.gz`
1. `conda activate lidargen`
1. `python main.py --config kitti.yml --exp kitti_pretrained --sample --ni`
1. `python visualization.py --exp kitti_pretrained`

A video similar to the following will be produced: 

![visualization example](https://github.com/vzyrianov/lidar-project/blob/main/camera_ready/assets/visualization_output.gif)

## Densification MAE and IOU

To perform densification run:

1. `python lidargen.py --densification --exp kitti_pretrained --config kitti.yml`

The target range images will be outputted to kitti_pretrained/densification_target. The LiDARGen densification result images will be outputted to kitti_pretrained/densification_result.

To calculate MAE (Mean Average Error) run:

1. `python lidargen.py --mae --exp kitti_pretrained --config kitti.yml`

To calculate IOU run:

1. `python lidargen.py --iou --exp kitti_pretrained --config kitti.yml`


## Samples

### 14k Samples

We provide a collection of 14,375 samples for research purposes here: [14K LiDARGen Samples](https://uofi.app.box.com/s/o3fdyrgdrsq5t108zvryt9ehnl06nicd/file/994876813141). 

### Baselines and ECCV'22 Benchmark Samples

We provide the LiDARGen and baseline samples used for performing the FID evaluation in our paper. The FID scores have marginal differences from the paper due to the random sampling performed to make FID calculation for RangeNet++ features possible. 

For any new applications we recommend using the LiDARGen samples from the "20k Samples" section above (which were generated with a newer model). 

|  Method          |  FID Score    |
|------------------|---------------|
| pc\_ncsn         | 2116.1611     |
| pc\_projectedgan | 2188.4207     |
| pc\_gan          | 3016.7556     |
| pc\_vae          | 2298.7101     |

Samples are available at this [link](https://uofi.box.com/shared/static/kmfe6alnt9xvnxnsvmv648a5mo8lq0j5.gz). Instructions to generate FID scores follow (replace the string "pc_projectedgan" in each instruction with the folder you chose):

1. `curl -L https://uofi.box.com/shared/static/kmfe6alnt9xvnxnsvmv648a5mo8lq0j5.gz --output lidargen_eccv22_samples.tar.gz`
1. `tar -xzf lidargen_eccv22_samples.tar.gz`
1. `python lidargen.py --fid_pointcloud pc_projectedgan --folder_name pc_projectedgan`
1. `python lidargen.py --fid --fid_folder1 pc_projectedgan_fid`


## Additional Information

### Range Image .pth format

LiDARGen represents LiDAR readings in a range image format (i.e., a 360 degree depth map with an additional intensity layer). These range images are normalized into a `[0, 1]` range with the transform `range_normalized = (np.log2(range_unnormalized+1)) / 6`.

When sampling or densification is performed, LiDARGen generates .pth files. These are serialized PyTorch tensors which can be loaded with `range_im = torch.load('filename.pth')`. The shape of this tensor is typically `[2, 64, 1024]` where the dimension represent `[channel, height, width]`. The range image depth value can be converted into a meter format by running `range_unnormalized = torch.exp2(range_im[0]*6) - 1`


## Bibtex

```
@inproceedings{zyrianov2022learning,
  title={Learning to Generate Realistic LiDAR Point Cloud},
  author={Zyrianov, Vlas and Zhu, Xiyue and Wang, Shenlong},
  booktitle={ECCV},
  year={2022}
}
```
