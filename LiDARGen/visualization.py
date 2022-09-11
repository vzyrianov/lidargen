import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import glob
import open3d.visualization.rendering as rendering
import os
import argparse
import torch
import pdb

def visualize_tensor(image):

    lidar_range = image[0] # range
    depth_range = np.exp2(lidar_range*6)-1
    lidar_intensity = image[1] # intensity

    range_view = cv2.resize(np.concatenate((depth_range, lidar_intensity*60), axis = 0), (1024, 512))

    # convert range to euclidean xyz
    fov_up=3.0
    fov_down=-25.0
    fov_up = fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad
    W = 1024.0
    H = 64.0
    x, y = np.meshgrid(np.arange(0, W), np.arange(0, H))
    x *= 1/W
    y *= 1/H
    yaw = np.pi*(x * 2 - 1)
    pitch = (1.0 - y)*fov - abs(fov_down)
    yaw = yaw.flatten()
    pitch = pitch.flatten()
    depth = depth_range.flatten()
    intensity_color = plt.cm.inferno(lidar_intensity.flatten())
    pts = np.zeros((len(yaw), 3))
    pts[:, 0] =  np.cos(yaw) * np.cos(pitch) * depth
    pts[:, 1] =  -np.sin(yaw) * np.cos(pitch) * depth
    pts[:, 2] =  np.sin(pitch) * depth

    # mask out invalid points
    mask = np.logical_and(depth>0.5, depth < 63.0)
    xyz = pts[mask, :]
    color = intensity_color[mask, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(color)
    # o3d.visualization.draw_geometries([pcd])

    # offscreen rendering
    render = rendering.OffscreenRenderer(1024, 768, headless=True)
    mtl = rendering.MaterialRecord()
    mtl.base_color = [1, 1, 1, 1]
    mtl.point_size = 2
    mtl.shader = "defaultLit"
    render.scene.set_background([255, 255, 255, 255])
    render.scene.add_geometry("point cloud", pcd, mtl)
    render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (1, 1, 1))
    render.scene.scene.enable_sun_light(True)
    render.scene.camera.look_at([0, 0, 0], [0, 0, 30], [0, 1, 0])
    bev_img = render.render_to_image()
    render.setup_camera(60.0, [0, 0, 0], [0, 20, 10], [0, 0, 1])
    pts_img = render.render_to_image()
    return bev_img, pts_img

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp", help="Experiment containing samples generated with final_only: false", type=str, required=True)

    args = parser.parse_args()

    folder = '{exp}/image_samples/images/'.format(exp=args.exp)
    all_files = glob.glob(folder + '/*.pth')
    file_count = len(all_files)

    os.system('mkdir {exp}/image_videos'.format(exp=args.exp))
    
    samples = 0

    for i in range(0, file_count):
        tensor = torch.load(folder + '/samples_' + str(i) + '.pth').numpy()

        for j in range(0, tensor.shape[0]):
            samples = tensor.shape[0]

            bev_img, pts_img = visualize_tensor(tensor[j])
            o3d.io.write_image("{exp}/image_videos/bev_{j}_{i}.png".format(exp=args.exp, i=str(i), j=str(j)), bev_img, quality=9)
            o3d.io.write_image("{exp}/image_videos/pts_{j}_{i}.png".format(exp=args.exp, i=str(i), j=str(j)), pts_img, quality=9)
    
    #Repeat final sample for several frames
    repetitions = int(file_count/4)
    for j in range(0, samples):
        
        tensor = torch.load(folder + '/samples_' + str(file_count-1) + '.pth').numpy()

        bev_img, pts_img = visualize_tensor(tensor[j])


        for i in range(0, repetitions):
            o3d.io.write_image("{exp}/image_videos/bev_{j}_{i}.png".format(exp=args.exp, i=str(i+file_count-1), j=str(j)), bev_img, quality=9)
            o3d.io.write_image("{exp}/image_videos/pts_{j}_{i}.png".format(exp=args.exp, i=str(i+file_count-1), j=str(j)), pts_img, quality=9)
            
    

    for j in range(0, samples):
        os.system('ffmpeg -framerate 60 -i {exp}/image_videos/pts_{j}_%0d.png -start_number 1000 pts_{j}.mp4'.format(exp=args.exp, j=j))
        os.system('ffmpeg -framerate 60 -i {exp}/image_videos/bev_{j}_%0d.png -start_number 1000 bev_{j}.mp4'.format(exp=args.exp, j=j))

if __name__=='__main__':
   main()



