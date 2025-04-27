"""
guocheng.qian@outlook.com
"""
import os
import argparse
import numpy as np
import torch
from PIL import Image
import imageio
import json
import glob
import shutil
from loguru import logger
import subprocess
import torch
import torch.nn as nn
from tqdm import tqdm
# set environment variable
os.environ["QT_QPA_PLATFORM"] = "offscreen" # offscreen for COLMAP


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


class MetricRescaler(nn.Module):
    def __init__(self, max_frames=100):
        super(MetricRescaler, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # ref: https://github.com/YvanYin/Metric3D/blob/main/onnx/test_onnx.py
        self.depth_model = (
            torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True)
            .to(self.device)
            .eval()
        )
        self.depth_input_size = (616, 1064)  # for vit model
        self.depth_mean = (
            torch.tensor([123.675, 116.28, 103.53]).float()[None, ..., None, None]
        )
        self.depth_std = torch.tensor([58.395, 57.12, 57.375]).float()[None, ..., None, None]
        self.max_frames = max_frames

    @staticmethod
    # Function to project 3D points to 2D and get depths
    def points3d_to_depth(points_3d, camera_matrices, rotations, translations):
        """
        Projects 3D points onto the image plane using camera parameters.

        :param points_3d: 3D points without RGB color in world coordinates (Nx3).
        :param camera_matrices: Camera intrinsic matrices (Bx3x3).
        :param rotations: Camera rotation matrices (Bx3x3).
        :param translations: Camera translation vectors (Bx3).
        :return: Tuple of projected 2D points (Bx2xN) and depths (BxN).
        """
        points_3d_expanded = points_3d.unsqueeze(0).expand(rotations.size(0), -1, -1)
        transformed_points = torch.bmm(rotations, points_3d_expanded.transpose(1, 2)) + translations.unsqueeze(-1)
        projected_points = torch.bmm(camera_matrices, transformed_points)
        depths = projected_points[:, 2,:]
        valid = depths > 0
        projected_points[:, 0,:][valid] /= depths[valid]
        projected_points[:, 1,:][valid] /= depths[valid]
        return projected_points[:,:2], depths

    @staticmethod
    # Function to generate depth maps from 2D points and depths
    def points2d_to_depthmap(image_shape, points_2d, depths):
        """
        Generates depth maps from projected 2D points and their corresponding depths.

        :param image_shape: Shape of the image (B, height, width).
        :param points_2d: Projected 2D points (Bx2xN).
        :param depths: Depths of the points (BxN).
        :return: Depth maps (BxHxW).
        """
        device = depths.device
        B, height, width = image_shape
        depth_maps = torch.full((B, height, width), float("inf"), dtype=torch.float32, device=device)
        flat_u = points_2d[:, 0,:].to(torch.int64)
        flat_v = points_2d[:, 1,:].to(torch.int64)
        batch_indices = torch.arange(B, device=device).reshape(-1, 1) * height * width
        linear_indices = (flat_v * width + flat_u) + batch_indices
        valid = (flat_u >= 0) & (flat_u < width) & (flat_v >= 0) & (flat_v < height)
        flat_depth_maps = depth_maps.view(-1)
        valid_linear_indices = linear_indices[valid]
        valid_depths = depths[valid]
        # Scatter minimum depths (occulusion-aware)
        flat_depth_maps.scatter_reduce_(dim=0, index=valid_linear_indices, src=valid_depths, reduce="min")
        return depth_maps

    def points3d_to_depthmaps(self, points_3d, camera_matrices, rotations, translations, image_shape):
        height, width = image_shape
        # Project points and generate depth maps
        points_2d, depths = self.points3d_to_depth(points_3d, camera_matrices, rotations, translations)
        return self.points2d_to_depthmap([len(depths), height, width], points_2d, depths)

    @staticmethod
    # Function to perform near plane scaling
    def run_near_plane_scaling(depth_maps):
        """
        Scales the scene based on the near plane depth
        based on 4.3 Refining poses with bundle adjustment in https://arxiv.org/pdf/1805.09817
        """
        near_planes = []
        for depth_map in depth_maps:
            valid_mask = torch.isfinite(depth_map) & (depth_map > 0)
            valid_depths = depth_map[valid_mask]
            if len(valid_depths) > 0:
                valid_depths_np = valid_depths.cpu().numpy()
                near_plane = np.percentile(valid_depths_np, 5)
                near_planes.append(near_plane)
        return 1.0 / np.percentile(near_planes, 10) # rescale the depth to 1.0m

    @staticmethod
    def convert_colmap_to_txt(sparse_dir):
        subprocess.run(
            [
                "colmap",
                "model_converter",
                "--input_path",
                sparse_dir,
                "--output_path",
                sparse_dir,  # Save txt files in the same folder
                "--output_type",
                "TXT",
            ]
        )

    @staticmethod
    # Function to read camera parameters from COLMAP text files
    def read_cameras_text(path):
        """
        Reads camera intrinsics from a COLMAP cameras.txt file.
        """
        with open(path, "r") as f:
            lines = f.readlines()
        cameras = {}
        for line in lines:
            if line.startswith("#") or line == "\n":
                continue
            data = line.strip().split()
            camera_id = int(data[0])
            params = list(map(float, data[4:]))
            cameras[camera_id] = np.array(params)
        return cameras

    @staticmethod
    # Function to read image poses from COLMAP text files
    def read_images_text(path):
        """
        Reads camera poses from a COLMAP images.txt file.
        """
        with open(path, "r") as f:
            lines = f.readlines()
        images = {}
        for i in range(0, len(lines), 2): # per 2 lines for each image
            if lines[i][0] == "#" or lines[i] == "\n":
                continue
            data = lines[i].strip().split()
            if len(data) > 0:
                image_id = int(data[0])
                qvec = np.array(list(map(float, data[1:5])))
                tvec = np.array(list(map(float, data[5:8])))
                camera_id = int(data[8])
                image_name = data[9]
                images[image_id] = (qvec, tvec, camera_id, image_name)
        return images

    @staticmethod
    # Function to read 3D points from COLMAP text files
    def read_points3D_text(path):
        """
        Reads 3D points from a COLMAP points3D.txt file.
        """
        with open(path, "r") as f:
            lines = f.readlines()
        points3D = {}
        for line in lines:
            if line.startswith("#") or line == "\n":
                continue
            data = line.strip().split()
            point_id = int(data[0])
            xyz = np.array(list(map(float, data[1:4])))
            rgb = np.array(list(map(float, data[4:7])), dtype=np.uint8)
            points3D[point_id] = (xyz, rgb)
        return points3D

    @staticmethod
    def save_camera_extrinsics_and_intrinsics(
        img_wh, camera_params, cam_w2cs, image_folder, sparse_dir
    ):
        os.makedirs(os.path.join(sparse_dir, "0"), exist_ok=True)
        # Save intrinsics to cameras.txt
        cameras_txt_path = os.path.join(sparse_dir, "cameras.txt")
        with open(cameras_txt_path, "w") as f:
            # Assuming single camera setup (ID 1) with Pinhole model
            f.write(
                "1 PINHOLE {} {} {} {} {} {}\n".format(
                    img_wh[0],
                    img_wh[1],
                    camera_params[0],
                    camera_params[1],
                    camera_params[2],
                    camera_params[3],
                )
            )

        # Save empty points3D.txt
        points3D_txt_path = os.path.join(sparse_dir, "points3D.txt")
        open(points3D_txt_path, "w").close()  # Empty points3D.txt

        # Save extrinsics (w2c) to images.txt
        images_txt_path = os.path.join(sparse_dir, "images.txt")
        with open(images_txt_path, "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_PATH\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            for i, w2c in enumerate(cam_w2cs):
                try:
                    image_path = os.path.join(image_folder, f"frame_{i:04d}.jpg")
                    Image.open(image_path) # check if image broken
                except:
                    logger.info(f"Image {image_path} is broken, skipping")
                    continue
                image_path = f"frame_{i:04d}.jpg"

                R = w2c[:3, :3]
                T = w2c[:3, 3]
                qvec = rotmat2qvec(R)
                camera_id = 1  # Assuming single camera

                # Write the first line (image info)
                f.write(
                    f"{i+1} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {T[0]} {T[1]} {T[2]} {camera_id} {image_path}\n"
                )

                # Leave the second line empty as required
                f.write("\n")

    @staticmethod
    def run_metric_scaling(depth_maps, target_depths, near_plane_scale=1.0, device="cuda"):
        scale = torch.nn.Parameter(torch.tensor(near_plane_scale, device=device))
        optimizer = torch.optim.Adam([scale], lr=0.01)
        criterion = torch.nn.L1Loss()

        valid_mask = (depth_maps != float("inf")) & (depth_maps > 0)
        valid_depth = depth_maps[valid_mask]
        target_depths = target_depths[valid_mask]

        valid_depth = valid_depth.to(device)
        target_depths = target_depths.to(device)
        # Set up tqdm progress bar
        pbar = tqdm(range(100), desc="Optimizing Scale")
        for _ in pbar:
            optimizer.zero_grad()
            scaled_depths = (
                valid_depth * scale
            )  # Scale the valid depths by the learned scale factor
            loss = criterion(scaled_depths, target_depths)
            loss.backward()
            optimizer.step()

            # Update progress bar with current loss value
            pbar.set_postfix({"loss": loss.item()})

        return scale.item(), torch.var(scaled_depths - target_depths).item()


    def __call__(self, video_path, new_camera_path=None,  camera_path=None, colmap_dir=None, image_folder="images"):
        """
        Scales a video based on COLMAP sparse reconstruction.

        :param args: Arguments for processing the video.
        """
        video_name = os.path.basename(video_path)
        video_dir = os.path.dirname(video_path)

        new_camera_path = os.path.join(video_dir, video_name.split('.')[0] + ".camera_info_metric_rescaled.json") \
            if new_camera_path is None else new_camera_path
        camera_path = os.path.join(video_dir, video_name.split('.')[0] + ".camera_info.json") \
            if camera_path is None else camera_path

        if os.path.isfile(new_camera_path):
            logger.info(f"Camera info already exists: {new_camera_path}")
            return

        frames = imageio.mimread(video_path, memtest=False)
        height, width = frames[0].shape[0:2]

        colmap_dir = os.path.join(video_dir, "colmap") if colmap_dir is None else colmap_dir
        colmap_output_dir = sparse_dir = os.path.join(colmap_dir, "sparse")
        os.makedirs(sparse_dir, exist_ok=True)

        if os.path.isfile(camera_path):
            triangulated_dir = os.path.join(colmap_dir, "triangulated")
            os.makedirs(triangulated_dir, exist_ok=True)
            colmap_output_dir = triangulated_dir
            # if original camera path is defined, load the camera info
            with open(camera_path, "r") as f:
                camera_info = json.load(f)

        # ============== Run Colmap to get Points3D ==============
        if not os.listdir(colmap_output_dir):
            # logger.warning(f"COLMAP folder not
            if not os.path.exists(camera_path):
                logger.warning(f"Camera info not found: {camera_path}, will run COLMAP with unknown poses")
                camera_info = None
            else:
                cam_intrinsics = np.asarray(camera_info["intrinsics"])
                cam_w2cs = np.asarray(camera_info["w2c"])
                intrinsics = [
                    cam_intrinsics[0][0, 0],
                    cam_intrinsics[0][1, 1],
                    cam_intrinsics[0][0, 2],
                    cam_intrinsics[0][1, 2],
                ]
                # fmt: off
                camera_params = f"{cam_intrinsics[0][0, 0]},{cam_intrinsics[0][1, 1]},{cam_intrinsics[0][0, 2]},{cam_intrinsics[0][1, 2]}"
                # fmt: on
                assert len(frames) >= len(cam_w2cs), "Number of frames should be no less than to the number of camera poses"
                if len(frames) > len(cam_w2cs):
                    logger.warning(
                        f"Number of frames in the video does not match the number of camera poses, use the first {len(cam_w2cs)} frames"
                    )
                    frames = frames[:len(cam_w2cs)]

            database_path = os.path.join(colmap_dir, "database.db")
            depth_dir = os.path.join(colmap_dir, "depth")
            os.makedirs(depth_dir, exist_ok=True)

            img_dir = os.path.join(video_dir, image_folder) #-----#
            image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
            if camera_info is not None and len(image_paths) > 0 and len(image_paths) != len(cam_w2cs):
                logger.warning(f"Number of frames in the video does not match the number of camera poses")
                shutil.rmtree(img_dir)
                image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

            if len(image_paths) == 0:
                logger.info(f"Extracting frames from video: {video_path}")
                os.makedirs(img_dir, exist_ok=True)
                for i, frame in enumerate(frames):
                    Image.fromarray(frame).save(os.path.join(img_dir, f"frame_{i:04d}.jpg"))

            ###################### Run COLMAP ######################
            extraction_command = [
                "colmap",
                "feature_extractor",
                "--database_path",
                database_path,
                "--image_path",
                img_dir,
                "--ImageReader.camera_model",
                "PINHOLE",
                "--ImageReader.single_camera",
                "1",
            ]
            if camera_info is not None:
                extraction_command.extend(["--ImageReader.camera_params", camera_params])

            # Step 1: Feature Extraction
            subprocess.run(extraction_command, check=True)

            # Step 2: Feature Matching
            subprocess.run(
                [
                    "colmap",
                    "sequential_matcher",  # for video
                    "--database_path",
                    database_path,
                ],
                check=True,
            )

            if camera_info is not None:
                self.save_camera_extrinsics_and_intrinsics(
                    (width, height), intrinsics, cam_w2cs, img_dir, sparse_dir
                )
                # Step 3: Point Triangulation
                subprocess.run(
                    [
                        "colmap",
                        "point_triangulator",
                        "--database_path",
                        database_path,
                        "--image_path",
                        img_dir,
                        "--input_path",
                        sparse_dir,
                        "--output_path",
                        triangulated_dir,
                    ],
                    check=True,
                )
                logger.info("Sparse reconstruction with known poses completed!")
            else:
                # Step 3: Sparse Reconstruction (Mapping)
                mapping_command = [
                    "colmap",
                    "mapper",
                    "--database_path",
                    database_path,
                    "--image_path",
                    img_dir,
                    "--output_path",
                    sparse_dir,
                ]
                if camera_info is not None:
                    mapping_command.extend(
                        [
                            "--Mapper.ba_refine_focal_length",
                            "0",
                            "--Mapper.ba_refine_principal_point",
                            "0",
                            "--Mapper.ba_refine_extra_param",
                            "0",
                        ]
                    )
                subprocess.run(mapping_command, check=True)
                logger.info("Sparse reconstruction with unknown poses completed!")
                if len(os.listdir(colmap_output_dir)) > 1:
                    logger.info("Scene as been split into multiple parts for registration, using scene id 0 for further calculations")
                colmap_output_dir = os.path.join(colmap_output_dir, "0")

        # Load COLMAP data
        self.convert_colmap_to_txt(colmap_output_dir)
        cameras = self.read_cameras_text(os.path.join(colmap_output_dir, "cameras.txt"))
        images = self.read_images_text(os.path.join(colmap_output_dir, "images.txt"))
        points3D_dict = self.read_points3D_text(os.path.join(colmap_output_dir, "points3D.txt"))
        points_3d = np.array([p for p in points3D_dict.values()]) # [N, 2, 3] xyz+rgb per point

        # Prepare camera parameters
        w2cs = []
        for image_id in images.keys():
            qvec, tvec, camera_id, _ = images[image_id]
            rotation = qvec2rotmat(qvec)
            translation = tvec
            w2c = np.eye(4)
            w2c[:3,:3] = rotation
            w2c[:3, 3] = translation
            w2cs.append(w2c)

        w2cs = np.stack(w2cs).astype(np.float32)
        camera = cameras[camera_id]
        fx, fy, cx, cy = camera[:4]

        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        rotations = torch.from_numpy(w2cs[:,:3,:3])
        translations = torch.from_numpy(w2cs[:,:3, 3])
        points_3d = torch.from_numpy(points_3d).float()
        camera_matrices = torch.from_numpy(camera_matrix).float().unsqueeze(0).repeat(len(images), 1, 1)

        # Project points and generate depth maps
        depth_maps = self.points3d_to_depthmaps(points_3d[:, 0], camera_matrices, rotations, translations, [height, width])

        # Perform near plane scaling
        near_plane_scale = self.run_near_plane_scaling(depth_maps)
        logger.info(f"Near plane scale: {near_plane_scale}")

        ###################### Metric depth ######################
        if width < height:
            logger.warning("[Warning] Metric3D is trained for landscape frames")
        scale = min(self.depth_input_size[0] / height, self.depth_input_size[1] / width)
        intrinsic = [fx, fy, cx, cy]
        intrinsic = [
            intrinsic[0] * scale,
            intrinsic[1] * scale,
            intrinsic[2] * scale,
            intrinsic[3] * scale,
        ]

        #### normalize
        # padding to self.depth_input_size
        pad_h = self.depth_input_size[0] - height
        pad_w = self.depth_input_size[1] - width
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        frames_tensor = torch.from_numpy(np.stack(frames)[:self.max_frames].transpose(0, 3, 1, 2)).float()  # Convert to tensor (B, C, H, W)
        resized_frames_tensor = torch.nn.functional.interpolate(
            frames_tensor,
            size=(int(height * scale), int(width * scale)),
            mode="bilinear",
            align_corners=False
        )
        resized_frames_tensor = (resized_frames_tensor - self.depth_mean) / self.depth_std
        rgb = torch.nn.functional.pad(
            resized_frames_tensor,
            pad=(pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half),
            mode="constant",
            value=0
        ).to(self.device)

        try:
            with torch.no_grad():
                pred_depth, _, _ = self.depth_model.inference({"input": rgb})
        except Exception as e:
            logger.error(f'Failed to get depth for {video_path} with error: {e}')
            return False

        # unpad
        pred_depth = pred_depth[
            :,
            :,
            pad_info[0] : pred_depth.shape[-2] - pad_info[1],
            pad_info[2] : pred_depth.shape[-1] - pad_info[3],
        ]
        # upsample to original size
        pred_depth = torch.nn.functional.interpolate(
            pred_depth, [height, width], mode="bilinear"
        ).squeeze()
        #### de-canonical transform
        canonical_to_real_scale = (
            intrinsic[0] / 1000.0
        )  # 1000.0 is the focal length of canonical camera
        pred_depth = pred_depth * canonical_to_real_scale  # now the depth is metric
        pred_depth = torch.clamp(pred_depth, 0, 300)

        metric_scale, metric_var = self.run_metric_scaling(
            depth_maps[:self.max_frames], pred_depth, near_plane_scale, device=self.device
        )
        logger.info(f"Metric scale: {metric_scale}")

        # rescale the w2cs
        metric_w2cs = w2cs.copy()
        metric_w2cs[:, :3, 3] *= metric_scale
        camera_info["w2c"] = [w2c[:3].tolist() for w2c in metric_w2cs]
        camera_info["scale_var"] = metric_var
        camera_info["metric_scale"] = metric_scale
        with open(new_camera_path, "w") as f:
            json.dump(camera_info, f, indent=4)
        logger.info(f"Camera info saved to {new_camera_path}")

def check_if_exists(path):
    return os.path.exists(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale video based on COLMAP sparse reconstruction")
    parser.add_argument("--video_path",
                        "-v",
                        type=str,
                        default='demo/0a5070800c721f85/0a5070800c721f85.mp4',
                        help="Path to the video file")
    parser.add_argument("--camera_path",
                        "-c",
                        type=str,
                        default=None,
                        help="Path to the camera pose file. Set to None if no precomputed camera poses")
    parser.add_argument("--output_camera_path",
                        "-o",
                        type=str,
                        default=None,
                        help="Path to the camera pose file. Set to None to save to the same folder as the video_path")
    parser.add_argument("--colmap_path",
                        "-co",
                        type=str,
                        default=None,
                        help="Path to the COLMAP output folder. Set to None if no precomputed colmap")
    args = parser.parse_args()

    scaler = MetricRescaler()
    scaler(args.video_path, args.output_camera_path, args.camera_path, args.colmap_path)
