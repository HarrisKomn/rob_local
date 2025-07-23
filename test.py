import sys
from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import glob
import argparse
from droid_slam.droid import Droid
from natsort import natsorted
import json

sys.path.append('droid_slam')


ply_header = '''ply
   format ascii 1.0
   element vertex %(vert_num)d
   property float x
   property float y
   property float z
   property uchar red
   property uchar green
   property uchar blue
   end_header
 '''

def enhance_image(image):

    # Create the sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # Sharpen the image
    sharpened_image = cv2.filter2D(image, -1, kernel)

    # Convert the image to grayscale to determine brightness
    gray = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)

    # Define the threshold for brightness
    threshold = 180

    # Create a mask for pixels brighter than the threshold
    bright_mask = gray > threshold

    # Set bright pixels to black in the original image
    sharpened_image[bright_mask] = [0, 0, 0]

    # Step 7: Display the original and enhanced images
    cv2.imshow('Original Image', image)
    cv2.imshow('Enhanced Image', sharpened_image)
    cv2.waitKey(1)

    return image

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def image_stream(imagedir, calib, stride=1):

    with open(calib, "r") as file:
        data = json.load(file)

    # Convert to NumPy arrays
    camera_matrix = np.array(data["camera_matrix"])
    dist_coeffs = np.array(data["dist_coeffs"])

    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
    intrinsics_vec = torch.as_tensor([fx, fy, cx, cy])


    ht0, wd0 = [400, 400]
    h1 = int(ht0 * np.sqrt((384 * 512) / (ht0 * wd0)))
    w1 = int(wd0 * np.sqrt((384 * 512) / (ht0 * wd0)))

    # read all png images in folder
    images_left = sorted(glob.glob(os.path.join(imagedir, 'left/*.png')))[::stride]
    images_right = [x.replace('left', 'right') for x in images_left]

    for t, (imgL, imgR) in enumerate(zip(images_left, images_right)):

        print("Image path: ", imgL)
        image_L = cv2.imread(imgL)

        rows, cols,_ = image_L.shape
        if rows == 400 and cols == 400:
            cropped_imgL = image_L
        else:
            crop_size = min(rows,cols,400)
            crop_x = (cols-crop_size)//2
            crop_y = (rows-crop_size)//2
            cropped_imgL = image_L[crop_y-1:crop_y+crop_size-1,crop_x+2:crop_x+crop_size+2]

        cropped_imgL = cv2.undistort(cropped_imgL, camera_matrix, dist_coeffs)
        cropped_imgL = enhance_image(cropped_imgL)

        cv2.imshow('Undistort Image', cropped_imgL)
        cv2.waitKey(1)

        images = [cropped_imgL]

        images = np.stack(images, 0)
        images = torch.as_tensor(images).permute(0, 3, 1, 2).to("cuda:0", dtype=torch.float32)
        images = images[:, :, :h1 - h1 % 8, :w1 - w1 % 8]

        intrinsics = torch.as_tensor(intrinsics_vec).cuda()
        intrinsics[0] *= w1/ wd0
        intrinsics[1] *= h1/ ht0
        intrinsics[2] *= w1/ wd0
        intrinsics[3] *= h1/ ht0

        yield stride * t, images, intrinsics

def save_reconstruction(droid, reconstruction_path):

    from pathlib import Path

    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    Path("{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("{}/images.npy".format(reconstruction_path), images)
    np.save("{}/disps.npy".format(reconstruction_path), disps)
    np.save("{}/poses.npy".format(reconstruction_path), poses)
    np.save("{}/intrinsics.npy".format(reconstruction_path), intrinsics)


if __name__ == '__main__':



    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[400, 400])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=1.5, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=1.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=8.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=16, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=20.0) #hk it was 3
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()
    args.stereo = False

    torch.multiprocessing.set_start_method('spawn')

    droid = None
    # rec_name = args.imagedir.split('/')[6] + '_'+ args.imagedir.split('/')[7]
    # #seq_id = args.imagedir.split('/')[7]
    # seq_id = args.imagedir.split('/')[-2] #RESULTS FOR 05-10-2024

    #args.seq_id = seq_id
    #save_dir = f"reconstructions/DELIVERABLE_NEW/{seq_id}/"

    #save_dir = args.reconstruction_path
    args.rec_path = args.reconstruction_path

    if not os.path.exists(args.rec_path):
        os.makedirs(args.rec_path)
        print(f"Directory '{args.rec_path}' created.")
    else:
        print(f"Directory '{args.rec_path}' already exists.")

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    args.upsample = True
    tstamps = []

    for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, calib=args.calib, stride=args.stride)):

        if t < args.t0:
            continue

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)

        droid.track(t, image, intrinsics=intrinsics)

    traj_est = droid.terminate(image_stream(args.imagedir, calib=args.calib,  stride=args.stride))

    np.save(f"{args.reconstruction_path}/SLAM_traj.npy", traj_est)

    print("====== Trajectories saved")

    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path)
        print("====== Reconstruction completed")


