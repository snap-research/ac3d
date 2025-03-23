import numpy as np
import torch
import math
import torch.nn.functional as F
import json

import dreifus
from dreifus.matrix import TorchPose
from dreifus.matrix import Pose, CameraCoordinateConvention, PoseType
from scipy.signal import savgol_filter

CAMERA = {
    
    "static": {     "angle":[0., 0., 0.],   "T":[0., 0., 0.]},
    "up": {     "angle":[0., 0., 0.],   "T":[0., 1., 0.]},
    "down": {   "angle":[0., 0., 0.],   "T":[0.,-1.,0.]},
    "left": {   "angle":[0., 0., 0.],   "T":[1.,0.,0.]},
    "right": {  "angle":[0., 0., 0.],   "T": [-1.,0.,0.]},
    "zoom_in": {    "angle":[0., 0., 0.],   "T": [0.,0.,-1.]},
    "zoom_out": {   "angle":[0., 0., 0.],   "T": [0.,0.,1.]},
    "ACW": {        "angle": [0., 0., 1.],  "T":[0., 0., 0.]},
    "CW": {         "angle": [0., 0., -1.], "T":[0., 0., 0.]},
    "ACW_x": {        "angle": [1., 0., 0.],  "T":[0., 0., 0.]},
    "CW_x": {         "angle": [-1., 0., 0.], "T":[0., 0., 0.]},
    "ACW_y": {        "angle": [0., 1., 0.],  "T":[0., 0., 0.]},
    "CW_y": {         "angle": [0., -1., 0.], "T":[0., 0., 0.]},
    
    "zoom_out_05_up_05": {"multi_traj": True, "angle":[0., 0., 0.], "T": [[0.0, 0.5, [0.,0.,1.]], [0.5, 1.0, [0.,1.,0.]]]},
    "zoom_in_05_down_05": {"multi_traj": True, "angle":[0., 0., 0.], "T": [[0.0, 0.5, [0.,0.,-1.]], [0.5, 1.0, [0.,-1.,0.]]]},
    "left_05_up_05": {"multi_traj": True, "angle":[0., 0., 0.], "T": [[0.0, 0.5, [1.,0.,0.]], [0.5, 1.0, [0.,1.,0.]]]},
    "right_05_left_05": {"multi_traj": True, "angle":[0., 0., 0.], "T": [[0.0, 0.5, [-1.,0.,0.]], [0.5, 1.0, [1.,0.,0.]]]},
    "zoom_out_05_left_05": {"multi_traj": True, "angle":[0., 0., 0.], "T": [[0.0, 0.5, [0.,0.,1.]], [0.5, 1.0, [1.,0.,0.]]]},
    "zoom_out_05_right_05": {"multi_traj": True, "angle":[0., 0., 0.], "T": [[0.0, 0.5, [0.,0.,1.]], [0.5, 1.0, [-1.,0.,0.]]]},
    "zoom_out_5.0_05_up_1.0_05": {"multi_traj": True, "angle":[0., 0., 0.], "T": [[0.0, 0.5, [0.,0.,5.]], [0.5, 1.0, [0.,1.,0.]]]},
    "zoom_out_3.0_05_right_1.5_05": {"multi_traj": True, "angle":[0., 0., 0.], "T": [[0.0, 0.5, [0.,0.,3.]], [0.5, 1.0, [-1.5,0.,0.]]]},
    
    "zoom_out_05_ACW_y_05": {"multi_traj": True, "multi_ang": True, "angle":[[0., 0., 0.], [0., 1., 0.]], "T": [[0.0, 0.5, [0.,0.,1.]], [0.5, 1.0, [0.,0.,0.]]]},
    "zoom_out_05_CW_y_05": {"multi_traj": True, "multi_ang": True, "angle":[[0., 0., 0.], [0., -1., 0.]], "T": [[0.0, 0.5, [0.,0.,1.]], [0.5, 1.0, [0.,0.,0.]]]},
    "zoom_in_05_ACW_y_05": {"multi_traj": True, "multi_ang": True, "angle":[[0., 0., 0.], [0., 1., 0.]], "T": [[0.0, 0.5, [0.,0.,-1.]], [0.5, 1.0, [0.,0.,0.]]]},
    "zoom_in_05_CW_y_05": {"multi_traj": True, "multi_ang": True, "angle":[[0., 0., 0.], [0., -1., 0.]], "T": [[0.0, 0.5, [0.,0.,-1.]], [0.5, 1.0, [0.,0.,0.]]]},
    
    "left_05_ACW_y_05": {"multi_traj": True, "multi_ang": True, "angle":[[0., 0., 0.], [0., 1., 0.]], "T": [[0.0, 0.5, [1.,0.,0.]], [0.5, 1.0, [0.,0.,0.]]]},
    "left_05_CW_y_05": {"multi_traj": True, "multi_ang": True, "angle":[[0., 0., 0.], [0., -1., 0.]], "T": [[0.0, 0.5, [1.,0.,0.]], [0.5, 1.0, [0.,0.,0.]]]},
    "right_05_ACW_y_05": {"multi_traj": True, "multi_ang": True, "angle":[[0., 0., 0.], [0., 1., 0.]], "T": [[0.0, 0.5, [-1.,0.,0.]], [0.5, 1.0, [0.,0.,0.]]]},
    "right_05_CW_y_05": {"multi_traj": True, "multi_ang": True, "angle":[[0., 0., 0.], [0., -1., 0.]], "T": [[0.0, 0.5, [-1.,0.,0.]], [0.5, 1.0, [0.,0.,0.]]]},
    
    "up_05_ACW_y_05": {"multi_traj": True, "multi_ang": True, "angle":[[0., 0., 0.], [0., 1., 0.]], "T": [[0.0, 0.5, [0.,1.,0.]], [0.5, 1.0, [0.,0.,0.]]]},
    "up_05_CW_y_05": {"multi_traj": True, "multi_ang": True, "angle":[[0., 0., 0.], [0., -1., 0.]], "T": [[0.0, 0.5, [0.,1.,0.]], [0.5, 1.0, [0.,0.,0.]]]},
    "down_05_ACW_y_05": {"multi_traj": True, "multi_ang": True, "angle":[[0., 0., 0.], [0., 1., 0.]], "T": [[0.0, 0.5, [0.,-1.,0.]], [0.5, 1.0, [0.,0.,0.]]]},
    "down_05_CW_y_05": {"multi_traj": True, "multi_ang": True, "angle":[[0., 0., 0.], [0., -1., 0.]], "T": [[0.0, 0.5, [0.,-1.,0.]], [0.5, 1.0, [0.,0.,0.]]]},
}


def rotate_around(num_frames, azimuth_start=90., azimuth_end=270., eval_elevation_deg=0., eval_camera_distance=9.0, camera_center=(0.0, 0.0, 0.0), offset_z=None, offset_x=1.0, offset_y=1.0):
    azimuth_deg = torch.linspace(azimuth_start, azimuth_end, num_frames)
    elevation_deg = torch.full_like(
        azimuth_deg, eval_elevation_deg
    )
    camera_distances = torch.full_like(
        elevation_deg, eval_camera_distance
    )

    elevation = elevation_deg * math.pi / 180
    azimuth = azimuth_deg * math.pi / 180

    # convert spherical coordinates to cartesian coordinates
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
        ],
        dim=-1,
    )

    # default scene center at origin
    center = torch.tensor(camera_center)[None].repeat(num_frames, 1)
    up = torch.as_tensor([0, 1, 0], dtype=torch.float32)[
        None, :
    ].repeat(1, 1)

    lookat = -F.normalize(center - camera_positions, dim=-1)
    right = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    w2c = convert_camera(c2w3x4)
    return w2c

def compute_R_form_rad_angle(angles):
    theta_x, theta_y, theta_z = angles
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def get_translation_time(perc, T):
    for idx, (perc_start, perc_end, T_sub) in enumerate(T):
        if perc >= perc_start and perc <= perc_end:
            T_final = T_sub
            break
    T_final = np.array(T_final).reshape(3,1)*(perc-perc_start)
    if idx != 0:
        T_final = T_final + np.array(T[idx-1][-1]).reshape(3,1)*T[idx-1][1]
    return T_final

def circle_traj_trans(frames_count, radius=0.25, angles=[0., 0., 0.], radius_y=None, reverse_dir=False, spiral=False, num_circles=1, r_start=0.0, non_linear_val=None):
    if radius_y is None:
        radius_y = radius
    if non_linear_val is not None:
        f = lambda x: x**non_linear_val
        angle = f(np.linspace(0, 1, frames_count))
    else:
        angle = np.linspace(0, 1, frames_count)
    angle *= num_circles*2*np.pi
    if spiral:
        rad_factor = np.linspace(0, 1, frames_count)
        radius = rad_factor*radius - r_start
        radius_y = rad_factor*radius_y - r_start
    x = np.cos(angle) * radius
    y = np.sin(angle) * radius_y
    z = np.zeros_like(x)
    T = np.stack([x,y,z],1)[..., None]
    R = compute_R_form_rad_angle(angles)[None].repeat(frames_count, 0)
    RT = np.concatenate([R,T], axis=2)
    if reverse_dir:
        RT = np.flip(RT, 0)
    return RT

def get_camera_motion(
    cam_type, frames_count, base_angle_deg = 90., base_T_norm=1.5, speed=1.0, rotate_around_kwargs={},
    out_tensor=False, device=None, circle_kwargs={}, window_size_savgol=0.25,
    ):
    if cam_type == "rotate_around":
        print("rotate_around_kwargs", rotate_around_kwargs)
        RT = rotate_around(frames_count, **rotate_around_kwargs)
    elif cam_type == "circle_traj_trans":
        RT = circle_traj_trans(frames_count, **circle_kwargs)
        if out_tensor:
            RT = torch.tensor(RT.copy())
    else:
        cam_cfg = CAMERA[cam_type]
        angle, T = cam_cfg["angle"], cam_cfg["T"]
        angle = np.array(angle)
        RT = []
        for i in range(frames_count):
            if "multi_ang" in cam_cfg and cam_cfg["multi_ang"]:
                angle_idx = 0 if i/frames_count < 0.5 else 1
                angle_calc = angle[angle_idx]
            else:
                angle_calc = angle
            _angle = (i/frames_count)*speed*(base_angle_deg*np.pi/180)*angle_calc
            R = compute_R_form_rad_angle(_angle)
            if "multi_traj" in cam_cfg and cam_cfg["multi_traj"]:
                _T=get_translation_time(i/frames_count, T)*speed*base_T_norm
            else:
                T = np.array(T).reshape(3,1)
                _T=(i/frames_count)*speed*base_T_norm*T
            _RT = np.concatenate([R,_T], axis=1)
            RT.append(_RT)
        RT = np.stack(RT)
        if out_tensor:
            RT = torch.tensor(RT)
    if device is not None:
        RT = RT.to(device)
    return RT

def convert_camera(c2ws):
    extra_row = torch.tensor([0., 0., 0., 1.])[None, None].expand(c2ws.shape[0], -1, -1)
    c2ws = torch.cat((torch.tensor(c2ws), extra_row), 1).numpy()
    w2cs = []
    for c2w in c2ws:
        pose = Pose(c2w, pose_type=PoseType.CAM_2_WORLD)
        w2c = pose.change_pose_type(PoseType.WORLD_2_CAM, inplace=False).numpy()
        w2cs.append(w2c)
    w2cs = torch.tensor(w2cs)
    w2cs = w2cs[:, :3]
    return w2cs

def write_camera(w2c_list, camera_info_json_path):
    if isinstance(w2c_list, torch.Tensor):
        w2c_list = w2c_list.cpu().numpy()
    w2c_list = w2c_list.tolist()
    intrinsics = [
        [619.8889770507812, 0.0, 512.0],
        [0.0, 619.8890380859375, 288.0],
        [0.0, 0.0, 1.0]
        ]
    imH = 576
    imW = 1024
    intrinsic_list = [intrinsics for _ in range(len(w2c_list))]
    camera_info = {
        "moving_camera": True,
        "moving_object": False,
        "imH": imH,
        "imW": imW,
        "intrinsics": intrinsic_list,
        "w2c": w2c_list,
    }
    with open(camera_info_json_path, 'w') as f_json:
        json.dump(camera_info, f_json, indent=4)

if __name__ == "__main__":
    frames_count = 121
    # cam_type = "zoom_out_05_up_05"
    cam_type = "rotate_around"
    extrinsics = get_camera_motion(cam_type, frames_count)
    out_path_camera = "/path/to/camera.json"
    write_camera(extrinsics, out_path_camera)