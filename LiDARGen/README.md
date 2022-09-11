# Score-Based Generative Models for LIDAR Generation

This repo contains modification of the the official implementation for the paper [Improved Techniques for Training Score-Based Generative Models](http://arxiv.org/abs/2006.09011). 

## Running Experiments

### Dependencies

Run the following to install all necessary python packages for our code.

```bash
pip install -r requirements.txt
```

### Training

For example, we can train LiDAR Generation on full KITTI dataset by running the following

```bash
python main.py --config kitti.yml --doc kitti
```

Log files will be saved in `<exp>/logs/kitti`.

Single example overfitting could be done as follows:

```bash
python main.py --config lidar.yml --doc lidar
```
