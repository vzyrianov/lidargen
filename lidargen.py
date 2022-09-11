
import sys
sys.path.append('rangenetpp/lidar_bonnetal_master/train/tasks/semantic')
sys.path.append('rangenetpp/lidar_bonnetal_master/train/')
import metrics.fid.lidargen_fid as lidargen_fid
import metrics.histogram.mmd as mmd
import argparse
import os
import torch
import metrics.mae as lidargen_mae
import metrics.iou as lidargen_iou
import metrics.histogram.jsd as jsd
import rangenetpp.lidar_bonnetal_master.train.tasks.semantic.infer_lib as rangenetpp
import glob


def generate_kitti_fid(folder_fid, folder_segmentations, sample_count,  seed=0):

    # Get dump for KITTI
    os.system("rm -r {folder_segmentations}".format(folder_segmentations=folder_segmentations))
    os.system("rm -r {folder_fid}".format(folder_fid=folder_fid))
    os.system("mkdir {folder_segmentations}".format(folder_segmentations=folder_segmentations))
    os.system("mkdir {folder_fid}".format(folder_fid=folder_fid))

    rangenetpp.main('--dataset ignore --model rangenetpp/lidar_bonnetal_master/darknet53-1024/ --kitti --output_dir {folder_segmentations} --frd_dir {folder_fid} --kitti_count {kitti_count} --seed {seed}'.format(
        kitti_count=str(sample_count), seed=str(seed), folder_segmentations=folder_segmentations, folder_fid=folder_fid))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", help="Train model. Requires --config and --exp", action="store_true")

    parser.add_argument("--sample", help="Generate unconditional samples from model.", action="store_true")
    parser.add_argument("--densification", help="Generate densification experiment samples.", action="store_true")

    # Unconditional Stats
    parser.add_argument("--visualize_samples", help="Generate top down visualizations of generated samples")
    parser.add_argument("--fid", help="Run generated samples through RangeNet to get FID (KITTI only).", action="store_true")
    parser.add_argument("--mmd", help="Calculate MMD between samples and KITTI-360", action="store_true")
    parser.add_argument("--jsd", help="Caculate JSD between samples and KITTI-360", action="store_true")

    # Densification Stats
    parser.add_argument("--iou", help="Run RangeNet++ IOU downstream comparison of LiDARGen upsampling with nearest neighbor", action="store_true")
    parser.add_argument("--mae", help="Get MAE (Range Representation) for upsampling with LiDARGen, Bicubic, and Nearest Neighbors", action="store_true")

    # General
    parser.add_argument("--config", help="Config to be used for sampling.")
    parser.add_argument("--exp", help="The experiment name. If using pretrained model, set to provided folder name.")
    parser.add_argument("--samples", help="Number of samples", type=int, default=8)

    # Manually provide folders
    parser.add_argument("--fid_folder1", help="Manually provide folder1", type=str, default=None)
    parser.add_argument("--fid_folder2", help="Manually provide folder2", type=str, default=None)

    parser.add_argument("--folder_name", help="Folder name for manual generation", type=str, default=None)
    parser.add_argument("--fid_pointcloud", help="Dump features for folder of .npy point clouds into --folder_name", type=str, default=None)

    args = parser.parse_args()

    if ((args.config is None or args.exp is None) and (args.fid_folder1 is None) and (args.fid_pointcloud is None)):
        print('--config and --exp flags are required. See --help for more information')
        return

    if (args.train):
        # Train the model...

        os.system('python LiDARGen/main.py --ni --exp {exp} --config {cfg}'.format(
            exp=str(args.exp), cfg=str(args.config)))

    if (args.sample):
        # Do sampling

        desired_samples = args.samples
        current_index = 0
        seed = 0

        os.system('mkdir ' + str(args.exp) + '/unconditional_samples')
        os.system('python LiDARGen/main.py --ni --sample --exp ' + args.exp + ' --config ' + args.config + ' --seed ' + str(seed))

        sample_1 = torch.load(str(args.exp) + '/image_samples/images/samples.pth')
        count_per_sample = sample_1.shape[0]
        for i in range(0, count_per_sample):
            torch.save(sample_1[i], str(
                args.exp) + '/unconditional_samples/' + str(current_index) + '.pth')
            current_index = current_index + 1
        seed = seed + 1

        while(current_index < desired_samples):
            os.system('python LiDARGen/main.py --ni --sample --exp ' + args.exp + ' --config ' + args.config + ' --seed ' + str(seed))

            sample = torch.load(str(args.exp) + '/image_samples/images/samples.pth')
            count_per_sample = sample_1.shape[0]
            for i in range(0, count_per_sample):
                torch.save(sample[i], str(args.exp) + '/unconditional_samples/' + str(current_index) + '.pth')
                current_index = current_index + 1
            seed = seed + 1

    if (args.densification):
        # Densified sampling
        desired_samples = args.samples
        current_index = 0
        seed = 0

        os.system('rm -r ' + str(args.exp) + '/densification_result')
        os.system('rm -r ' + str(args.exp) + '/densification_target')

        os.system('mkdir ' + str(args.exp) + '/densification_result')
        os.system('mkdir ' + str(args.exp) + '/densification_target')
        os.system('python LiDARGen/main.py --ni --densification --sample --exp ' + args.exp + ' --config ' + args.config + ' --seed ' + str(seed))

        sample_result = torch.load(str(args.exp) + '/image_samples/images/densify_samples_result.pth')
        sample_target = torch.load(str(args.exp) + '/image_samples/images/densify_samples_target.pth')
        count_per_sample = sample_result.shape[0]
        for i in range(0, count_per_sample):
            torch.save(sample_result[i], str(args.exp) + '/densification_result/' + str(current_index) + '.pth')
            torch.save(sample_target[i], str(args.exp) + '/densification_target/' + str(current_index) + '.pth')
            current_index = current_index + 1
        seed = seed+1

        while(current_index < desired_samples):
            os.system('python LiDARGen/main.py --ni --densification --sample --exp ' +
                      args.exp + ' --config ' + args.config + ' --seed ' + str(seed))

            sample_result = torch.load(
                str(args.exp) + '/image_samples/images/densify_samples_result.pth')
            sample_target = torch.load(
                str(args.exp) + '/image_samples/images/densify_samples_target.pth')
            count_per_sample = sample_result.shape[0]
            for i in range(0, count_per_sample):
                torch.save(sample_result[i], str(args.exp) + '/densification_result/' + str(current_index) + '.pth')
                torch.save(sample_target[i], str(args.exp) + '/densification_target/' + str(current_index) + '.pth')
                current_index = current_index + 1
            seed = seed+1

    if(args.fid):
        folder1 = ""
        folder2 = ""

        if((not (args.fid_folder1 is None)) and (not (args.fid_folder2 is None))):
            folder1 = args.fid_folder1
            folder2 = args.fid_folder2

        elif (not (args.fid_folder1 is None)):
            folder1 = args.fid_folder1
            folder2 = "kitti_fid"
            folder_segmentations = "kitti_seg"
            generate_kitti_fid(folder2, folder_segmentations, 1000, 0)

        else:
            # Get dump for model samples
            os.system("rm -r {exp}/unconditional_fid".format(exp=args.exp))
            os.system("mkdir {exp}/unconditional_fid".format(exp=args.exp))
            os.system("rm -r {exp}/unconditional_segmentations".format(exp=args.exp))
            os.system("mkdir {exp}/unconditional_segmentations".format(exp=args.exp))

            rangenetpp.main('--dataset ignore --model rangenetpp/lidar_bonnetal_master/darknet53-1024/ --dump {exp}/unconditional_samples --output_dir {exp}/unconditional_segmentations --frd_dir {exp}/unconditional_fid'.format(exp=str(args.exp)))
            # Get dump for model samples

            sample_count = len(glob.glob("{exp}/unconditional_segmentations/*".format(exp=args.exp)))

            # Get dump for KITTI
            folder_kitti_seg = "kitti_segmentations"
            folder_kitti_fid = "kitti_fid"
            generate_kitti_fid(folder_kitti_fid, folder_kitti_seg, sample_count, 0)

            folder1 = "{exp}/unconditional_fid/".format(exp=str(args.exp))
            folder2 = folder_kitti_fid 

        fid_score = lidargen_fid.get_fid(folder1, folder2)

        print('-------------------------------------------------')
        print('FID Score: ' + str(fid_score))
        print('-------------------------------------------------')

    if (not (args.fid_pointcloud is None)):
        # Get dump for model samples
        os.system("rm -r {exp}_fid".format(exp=args.folder_name))
        os.system("mkdir {exp}_fid".format(exp=args.folder_name))
        os.system("rm -r {exp}_seg".format(exp=args.folder_name))
        os.system("mkdir {exp}_seg".format(exp=args.folder_name))

        rangenetpp.main('--dataset ignore --point_cloud --model rangenetpp/lidar_bonnetal_master/darknet53-1024/ --dump {pc_folder} --output_dir {exp}_seg --frd_dir {exp}_fid'.format(pc_folder=str(args.fid_pointcloud), exp=str(args.folder_name)))

    if (args.mmd):
        mmd_score = mmd.calculate_mmd("{exp}/unconditional_samples/".format(exp=str(args.exp)))

        print('-------------------------------------------------')
        print('MMD Score: ' + str(mmd_score))
        print('-------------------------------------------------')

    if (args.jsd):
        jsd_score = jsd.calculate_jsd("{exp}/unconditional_samples/".format(exp=str(args.exp)))

        print('-------------------------------------------------')
        print('JSD Score: ' + str(jsd_score))
        print('-------------------------------------------------')

    if(args.iou):
        os.system("rm -r " + str(args.exp) + '/target_rangenet_segmentations')
        os.system("rm -r " + str(args.exp) + '/target_rangenet_fid')
        os.system("mkdir " + str(args.exp) + "/target_rangenet_segmentations")
        os.system("mkdir " + str(args.exp) + "/target_rangenet_fid")

        os.system("rm -r " + str(args.exp) + '/result_rangenet_segmentations')
        os.system("rm -r " + str(args.exp) + '/result_rangenet_fid')
        os.system("mkdir " + str(args.exp) + "/result_rangenet_segmentations")
        os.system("mkdir " + str(args.exp) + "/result_rangenet_fid")

        rangenetpp.main("--dataset ignore --model rangenetpp/lidar_bonnetal_master/darknet53-1024/ --dump {exp}/densification_result/ --output_dir {exp}/result_rangenet_segmentations --frd_dir {exp}/result_rangenet_fid".format(exp=str(args.exp)))

        rangenetpp.main('--dataset ignore --model rangenetpp/lidar_bonnetal_master/darknet53-1024/ --dump {exp}/densification_target/ --output_dir {exp}/target_rangenet_segmentations --frd_dir {exp}/target_rangenet_fid'.format(exp=str(args.exp)))


        iou = lidargen_iou.calculate_iou("{exp}/result_rangenet_segmentations".format(exp=args.exp), "{exp}/target_rangenet_segmentations".format(exp=args.exp))

        print('-------------------------------------------------')
        print('IOU Score: ' + str(iou))
        print('-------------------------------------------------')

    if(args.mae):
        lidargen_mae.calculate_mae('./{exp}/'.format(exp=str(args.exp)))

    print('-------------------------------------------------')
    print('LiDARGen Completed. Enjoy')
    print('-------------------------------------------------')


if __name__ == '__main__':
    main()
