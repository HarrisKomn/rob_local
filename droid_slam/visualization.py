import torch
import cv2
import lietorch
import droid_backends
import time
import argparse
import numpy as np
import open3d as o3d
import time
from lietorch import SE3
import droid_slam.geom.projective_ops as pops

CAM_POINTS = np.array([
    [0, 0, 0],
    [-1, -1, 1.5],
    [1, -1, 1.5],
    [1, 1, 1.5],
    [-1, 1, 1.5],
    [-0.5, 1, 1.5],
    [0.5, 1, 1.5],
    [0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]])


def white_balance(img):
    # from https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def create_camera_actor(g, scale=0.01):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1 - g), 0.9 * (1 - g))
    camera_actor.paint_uniform_color(color)
    return camera_actor


def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    ###############################################################
    # HK ADDITION TO REMOVE BLACK PIXELS
    # Convert point cloud to numpy arrays
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    # Define a threshold for dark colors (you can adjust it based on your needs)
    # This example considers an RGB value of (0.1, 0.1, 0.1) as "very dark"
    dark_color_threshold = 0.2

    # Filter points with color above the dark threshold
    # (this will keep only those points that are not very dark)
    mask = (colors[:, 0] > dark_color_threshold) | \
           (colors[:, 1] > dark_color_threshold) | \
           (colors[:, 2] > dark_color_threshold)

    # Apply the mask to keep only non-dark points
    filtered_points = points[mask]
    filtered_colors = colors[mask]

    # Create a new point cloud with the filtered points and colors
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)
    ###############################################################

    return filtered_point_cloud


def droid_visualization(video, device="cuda:0"):
    """ DROID visualization frontend """

    torch.cuda.set_device(device)
    droid_visualization.video = video
    droid_visualization.cameras = {}
    droid_visualization.points = {}
    droid_visualization.warmup = 8
    droid_visualization.scale = 1.0
    droid_visualization.ix = 0


    #droid_visualization.filter_thresh = 0.005
    droid_visualization.filter_thresh = 0.009

    def increase_filter(vis):
        droid_visualization.filter_thresh *= 2
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True

    def decrease_filter(vis):
        droid_visualization.filter_thresh *= 0.5
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True

    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

        # hk# Additional code for point cloud saving
        # pts_result = np.empty([1,3])
        # clr_result = np.empty([1,3])

        with torch.no_grad():
            with video.get_lock():
                t = video.counter.value
                dirty_index, = torch.where(video.dirty.clone())
                dirty_index = dirty_index

            if len(dirty_index) == 0:
                return

            video.dirty[dirty_index] = False

            # convert poses to 4x4 matrix
            poses = torch.index_select(video.poses, 0, dirty_index)
            disps = torch.index_select(video.disps, 0, dirty_index)
            Ps = SE3(poses).inv().matrix().cpu().numpy()

            images = torch.index_select(video.images, 0, dirty_index)
            images = images.cpu()[:, [2, 1, 0], 3::8, 3::8].permute(0, 2, 3, 1) / 255.0
            points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()

            thresh = droid_visualization.filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))

            count = droid_backends.depth_filter(
                video.poses, video.disps, video.intrinsics[0], dirty_index, thresh)

            count = count.cpu()
            disps = disps.cpu()
            masks = ((count >= 2) & (disps > .5 * disps.mean(dim=[1, 2], keepdim=True)))

            for i in range(len(dirty_index)):
                pose = Ps[i]
                ix = dirty_index[i].item()

                if ix in droid_visualization.cameras:
                    vis.remove_geometry(droid_visualization.cameras[ix])
                    del droid_visualization.cameras[ix]

                if ix in droid_visualization.points:
                    vis.remove_geometry(droid_visualization.points[ix])
                    del droid_visualization.points[ix]

                ### add camera actor ###
                cam_actor = create_camera_actor(True)
                cam_actor.transform(pose)
                vis.add_geometry(cam_actor)
                droid_visualization.cameras[ix] = cam_actor

                mask = masks[i].reshape(-1)
                pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
                # print("====",pts[0])
                # pts[:,-1] = -pts[:,-1]
                # print("====",pts[0])

                # break
                clr = images[i].reshape(-1, 3)[mask].cpu().numpy()

                # print("===========",pts.shape)
                # hk# Additional code for point cloud saving
                # pts_result = np.append(pts_result, pts, axis=0)
                # clr_result = np.append(clr_result, clr, axis=0)

                ## add point actor ###
                point_actor = create_point_actor(pts, clr)
                vis.add_geometry(point_actor)
                droid_visualization.points[ix] = point_actor

            # # hk# Additional code for point cloud saving
            # result_point_actor = create_point_actor(pts_result, clr_result)
            # o3d.io.write_point_cloud(f'pcd_result.pcd', result_point_actor)

            ### Hack to save Point Cloud Data and Camera results ###

            # Save points
            pcd_points = o3d.geometry.PointCloud()
            for p in droid_visualization.points.items():
                pcd_points += p[1]
            o3d.io.write_point_cloud(f"{video.rec_path}/point_cloud.ply", pcd_points, write_ascii=False)
            #print("====== point cloud saved")


            # Save pose
            pcd_camera = create_camera_actor(True)
            for c in droid_visualization.cameras.items():
                pcd_camera += c[1]

            o3d.io.write_line_set(f"{video.rec_path}/camera_keyframes_traj.ply", pcd_camera, write_ascii=False)
            #print("====== point cloud for camera keyframes saved")

            ### end ###

            # hack to allow interacting with vizualization during inference
            if len(droid_visualization.cameras) >= droid_visualization.warmup:
                cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

            droid_visualization.ix += 1
            vis.poll_events()

            # time.sleep(3)

            vis.update_renderer()

    ### create Open3D visualization ###
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)
    vis.register_key_callback(ord("S"), increase_filter)
    vis.register_key_callback(ord("A"), decrease_filter)

    vis.create_window(height=540, width=960)

    vis.get_render_option().load_from_json("misc/renderoption.json")

    # vis.get_view_control()
    ctr = vis.get_view_control()
    print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
    ctr.change_field_of_view(100)
    print("======================================")
    print("Field of view (after changing) %.2f" % ctr.get_field_of_view())

    vis.run()
    vis.destroy_window()
# import torch
# import cv2
# import lietorch
# import droid_backends
# import time
# import argparse
# import numpy as np
# import open3d as o3d
# import time
# from lietorch import SE3
# import droid_slam.geom.projective_ops as pops
# import matplotlib.pyplot as plt
#
#
# CAM_POINTS = np.array([
#         [ 0,   0,   0],
#         [-1,  -1, 1.5],
#         [ 1,  -1, 1.5],
#         [ 1,   1, 1.5],
#         [-1,   1, 1.5],
#         [-0.5, 1, 1.5],
#         [ 0.5, 1, 1.5],
#         [ 0, 1.2, 1.5]])
#
# CAM_LINES = np.array([
#     [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])
#
# def white_balance(img):
#     # from https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
#     result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     avg_a = np.average(result[:, :, 1])
#     avg_b = np.average(result[:, :, 2])
#     result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
#     result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
#     result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
#     return result
#
# def create_camera_actor(g, scale=0.01):
#     """ build open3d camera polydata """
#     camera_actor = o3d.geometry.LineSet(
#         points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
#         lines=o3d.utility.Vector2iVector(CAM_LINES))
#
#     color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
#     camera_actor.paint_uniform_color(color)
#     return camera_actor
#
# def create_point_actor(points, colors):
#     """ open3d point cloud from numpy array """
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(points)
#     point_cloud.colors = o3d.utility.Vector3dVector(colors)
#     return point_cloud
#
# def set_initial_camera_view(vis):
#     view_control = vis.get_view_control()
#     view_control.set_front([0.4699189535486108, -0.62764022425775134, 0.62068021233921933])
#     view_control.set_lookat([-100, -100, 0])
#     view_control.set_up([-0.36828927493940194, 0.49961995188329117, 0.78405542766104697])
#     view_control.set_zoom(0.4)
# def droid_visualization(video, device="cuda:0"):
#     """ DROID visualization frontend """
#
#     torch.cuda.set_device(device)
#     droid_visualization.video = video
#     droid_visualization.cameras = {}
#     droid_visualization.points = {}
#     droid_visualization.warmup = 8
#     droid_visualization.scale = 1.0
#     droid_visualization.ix = 0
#
#     droid_visualization.filter_thresh = 0.01#0.005
#
#     def increase_filter(vis):
#         droid_visualization.filter_thresh *= 2
#         with droid_visualization.video.get_lock():
#             droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True
#
#     def decrease_filter(vis):
#         droid_visualization.filter_thresh *= 0.5
#         with droid_visualization.video.get_lock():
#             droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True
#
#     def animation_callback(vis):
#         view_control = vis.get_view_control()
#
#         view_control.set_front([0.4699189535486108, -0.62764022425775134, 0.62068021233921933])
#         view_control.set_lookat([0.97243714332580566, -0.1751408576965332, 0.51464511454105377])
#         view_control.set_up([-0.36828927493940194, 0.49961995188329117, 0.78405542766104697])
#         view_control.set_zoom(100.15999999999999961)
#
#
#         #hk# Additional code for point cloud saving
#         # pts_result = np.empty([1,3])
#         # clr_result = np.empty([1,3])
#         # vis.poll_events()
#         # vis.update_renderer()
#         with torch.no_grad():
#
#             with video.get_lock():
#                 t = video.counter.value
#                 dirty_index, = torch.where(video.dirty.clone())
#                 dirty_index = dirty_index
#
#             if len(dirty_index) == 0:
#                 return
#
#
#             video.dirty[dirty_index] = False
#
#             # convert poses to 4x4 matrix
#             poses = torch.index_select(video.poses, 0, dirty_index)
#             disps = torch.index_select(video.disps, 0, dirty_index)
#             Ps = SE3(poses).inv().matrix().cpu().numpy()
#
#             images = torch.index_select(video.images, 0, dirty_index)
#             images = images.cpu()[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
#             points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()
#
#
#             thresh = droid_visualization.filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))
#
#             count = droid_backends.depth_filter(
#                 video.poses, video.disps, video.intrinsics[0], dirty_index, thresh)
#
#             count = count.cpu()
#             disps = disps.cpu()
#             masks = ((count >= 2) & (disps > 0.5*disps.mean(dim=[1,2], keepdim=True))) # it was 0.5
#
#             for i in range(len(dirty_index)):
#
#                 # ctr = vis.get_view_control()
#                 # parameters = ctr.convert_to_pinhole_camera_parameters()
#
#                 pose = Ps[i]
#                 ix = dirty_index[i].item()
#
#                 if ix in droid_visualization.cameras:
#                     vis.remove_geometry(droid_visualization.cameras[ix])
#                     del droid_visualization.cameras[ix]
#
#                 if ix in droid_visualization.points:
#                     vis.remove_geometry(droid_visualization.points[ix])
#                     del droid_visualization.points[ix]
#
#                 ### add camera actor ###
#                 cam_actor = create_camera_actor(True)
#                 cam_actor.transform(pose)
#                 vis.add_geometry(cam_actor)
#                 droid_visualization.cameras[ix] = cam_actor
#
#                 mask = masks[i].reshape(-1)
#                 pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
#                 #print("====",pts[0])
#                 #pts[:,-1] = -pts[:,-1]
#                 #print("====",pts[0])
#
#                 #break
#                 clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
#
#                 #print("===========",pts.shape)
#                 #hk# Additional code for point cloud saving
#                 # pts_result = np.append(pts_result, pts, axis=0)
#                 # clr_result = np.append(clr_result, clr, axis=0)
#
#                 ## add point actor ###
#                 point_actor = create_point_actor(pts, clr)
#                 vis.add_geometry(point_actor)
#                 droid_visualization.points[ix] = point_actor
#
#                 # Restore camera parameters
#                 # ctr.convert_from_pinhole_camera_parameters(parameters)
#                 vis.poll_events()
#                 vis.update_renderer()
#
#
#             # # hk# Additional code for point cloud saving
#             # result_point_actor = create_point_actor(pts_result, clr_result)
#             # o3d.io.write_point_cloud(f'pcd_result.pcd', result_point_actor)
#
#             ### Hack to save Point Cloud Data and Camera results ###
#
#             print("==============Dirty point: ", dirty_index)
#             # Save points
#             pcd_points = o3d.geometry.PointCloud()
#             for p in droid_visualization.points.items():
#                 pcd_points += p[1]
#             o3d.io.write_point_cloud(f"points_test_25_05.ply", pcd_points, write_ascii=False)
#
#             # Save pose3
#             pcd_camera = create_camera_actor(True)
#             for c in droid_visualization.cameras.items():
#                 pcd_camera += c[1]
#
#             o3d.io.write_line_set(f"camera.ply", pcd_camera, write_ascii=False)
#
#             ### end ###
#
#
#             # hack to allow interacting with vizualization during inference
#             if len(droid_visualization.cameras) >= droid_visualization.warmup:
#                 cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
#                 #ctr.convert_from_pinhole_camera_parameters(parameters)
#
#
#
#             droid_visualization.ix += 1
#             vis.poll_events()
#             vis.update_renderer()
#
#
#
#     # vis = o3d.visualization.VisualizerWithKeyCallback()
#     # vis.register_animation_callback(animation_callback)
#     # vis.register_key_callback(ord("S"), increase_filter)
#     # vis.register_key_callback(ord("A"), decrease_filter)
#     # vis.create_window(height=540, width=960)
#     #
#     # # Set initial camera parameters
#     # camera_parameters = o3d.camera.PinholeCameraParameters()
#     # intrinsics = o3d.camera.PinholeCameraIntrinsic()
#     # intrinsics.set_intrinsics(960, 540, 800, 800, 480, 270)
#     # camera_parameters.intrinsic = intrinsics
#     #
#     # # Define extrinsics (translation and rotation)
#     # extrinsics = np.eye(4)
#     # extrinsics[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi / 4, 0))  # Rotate 45 degrees around Y axis
#     # extrinsics[:3, 3] = [0, 0, -3]  # Camera positioned 3 units back along the Z axis
#     # camera_parameters.extrinsic = extrinsics
#     #
#     # ctr = vis.get_view_control()
#     # ctr.convert_from_pinhole_camera_parameters(camera_parameters)
#     #
#     # set_initial_camera_view(vis)
#     # vis.run()
#     # vis.destroy_window()
#
#
#
#     ### create Open3D visualization ###
#     vis = o3d.visualization.VisualizerWithKeyCallback()
#     vis.register_animation_callback(animation_callback)
#     vis.register_key_callback(ord("S"), increase_filter)
#     vis.register_key_callback(ord("A"), decrease_filter)
#
#     vis.create_window(height=540, width=960)
#     vis.get_render_option().load_from_json("misc/renderoption.json")
#
#     # Set initial camera view and save it
#
#     # Set initial camera parameters
#     camera_parameters = o3d.camera.PinholeCameraParameters()
#     # Define intrinsics
#     intrinsics = o3d.camera.PinholeCameraIntrinsic()
#     intrinsics.set_intrinsics(960, 540, 800, 800, 480, 270)  # Adjust these values as needed
#
#     camera_parameters.intrinsic = intrinsics
#
#     # Define extrinsics (translation and rotation)
#     # Adjust the translation vector to set the camera position
#     # Adjust the rotation matrix to set the camera orientation
#     extrinsics = np.array([
#         [-0, 0, 0, 100],  # Rotation part (identity matrix for no rotation)
#         [0, -0, 0, 100],
#         [0, 0, -0, 100],  # Position the camera 3 units back along the Z axis
#         [0, 0, 0, 0]
#     ])
#     camera_parameters.extrinsic = extrinsics
#     ctr = vis.get_view_control()
#     ctr.convert_to_pinhole_camera_parameters()
#
#     # vis.get_view_control()
#     #ctr = vis.get_view_control()
#     #print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
#     #view_control = vis.get_view_control()
#
#     # view_control.set_front([0.4699189535486108, -0.62764022425775134, 0.62068021233921933])
#     # view_control.set_lookat([-0.97243714332580566, -0.1751408576965332, 0.51464511454105377])
#     # view_control.set_up([-0.36828927493940194, 0.49961995188329117, 0.78405542766104697])
#     # view_control.set_zoom(0.15999999999999961)
#
#
#     print("======================================")
#     #print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
#
#     vis.run()
#     vis.destroy_window()
