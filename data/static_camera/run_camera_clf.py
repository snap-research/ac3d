import os
from typing import List, Tuple
from collections import defaultdict

import numpy as np
import torch
import einops
from pytorchvideo.transforms import UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda, Resize
from tqdm import tqdm
import hydra
import PIL
from loguru import logger

import utils

#----------------------------------------------------------------------------

def detect_camera_type(video: torch.Tensor, detection_cfg) -> float:
    """
    Compute the average fraction of unchanged border pixels across consecutive frames.
    video: Tensor of shape (t, c, h, w). Assumed to be in a comparable range (e.g. [0,1]).
    border_size: Number of pixels to consider along each border.
    tol: Maximum allowed L1 difference (summed over channels) for a pixel to be considered unchanged.
    Returns: The average fraction (over frame transitions) of border pixels with L1 difference < tol.
    """
    t, c, h, w = video.shape
    # Create a border mask of shape (h, w): True for border pixels, False for interior.
    border_mask = torch.zeros((h, w), dtype=torch.bool)
    border_mask[:detection_cfg.border_size, :] = True
    border_mask[-detection_cfg.border_size:, :] = True
    border_mask[:, :detection_cfg.border_size] = True
    border_mask[:, -detection_cfg.border_size:] = True

    num_moved_frames = 0

    for i in range(1, t):
        border_prev, border_curr = video[i-1, :, border_mask], video[i, :, border_mask] # [c, num_border_pixels], [c, num_border_pixels]

        # Compute L1 norm difference per pixel (summing over channels) and the fraction of border pixels that did not change much
        diff_norm = torch.abs(border_curr - border_prev).sum(dim=0) # [num_border_pixels]
        moved_pixels_amount = (diff_norm > detection_cfg.tol).sum().item() # [1]
        moved_pixels_ratio = moved_pixels_amount / diff_norm.numel() # [1]

        # We should also check the colors std deviation of the border pixels
        # pixels_std_prev, pixels_std_curr = border_prev.std(dim=1).mean().item(), border_curr.std(dim=1).mean().item() # [1], [1]
        pixels_std_prev, pixels_std_curr = cluster_distance_metric(border_prev, detection_cfg.num_clusters), cluster_distance_metric(border_curr, detection_cfg.num_clusters) # [1], [1]
        is_uniform_color = max(pixels_std_prev, pixels_std_curr) < detection_cfg.min_cluster_dist
        is_much_movement = moved_pixels_ratio > (1.0 - detection_cfg.static_pixels_ratio_thresh)

        if is_uniform_color or is_much_movement:
            # If the colors do not change much, then we are very uncertain about our decision.
            # In this case, it is better to pretend like the camera is moving to maintain precision.
            num_moved_frames += 1

    prob_static = 1.0 if num_moved_frames == 0 else 0.0

    return prob_static

def cluster_distance_metric(pixels, num_clusters=3, max_iters=50) -> float:
    """
    Compute the mean squared distance of pixels to their assigned cluster centers from k-means clustering.
    """
    c, num_pixels = pixels.shape

    # Initialize cluster centers randomly from the data
    indices = torch.randperm(num_pixels)[:num_clusters]
    cluster_centers = pixels[:, indices] # [c, num_clusters]

    for _ in range(max_iters):
        distances = torch.cdist(pixels.T, cluster_centers.T) # [num_pixels, num_clusters]
        cluster_assignments = torch.argmin(distances, dim=1) # [num_pixels]
        new_centers = torch.stack([pixels[:, cluster_assignments == k].mean(dim=1) for k in range(num_clusters)], dim=1) # [c, num_clusters]
        if torch.allclose(cluster_centers, new_centers, atol=1e-6):
            break
        cluster_centers = new_centers

    # Compute mean squared distance to cluster centers
    assigned_centers = cluster_centers[:, cluster_assignments]  # [c, num_pixels]
    distances = torch.norm(pixels - assigned_centers, dim=0)  # [num_pixels]
    mean_distance = distances.mean().item()

    return mean_distance

#----------------------------------------------------------------------------
# Data processing.

def parse_video(video_file, clip_duration, num_frames, max_num_clips: int=4, ram_cache_dir: os.PathLike=None) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
    ram_cache_path = os.path.join(ram_cache_dir, f'{os.path.basename(video_file)}.pt') if ram_cache_dir is not None else None

    if os.path.isfile(ram_cache_path):
        # Trying to load from RAM cache
        try:
            state = torch.load(ram_cache_path)
            return state['video_tensors'], state['start_end_secs']
        except Exception as e:
            logger.error(f'Error: {e}')

    try:
        video_decoder = utils.VideoDecoder(video_file)
        duration = float(video_decoder.video_stream.duration * video_decoder.video_stream.time_base)
        if duration < 1:
            return [], []

        if duration < clip_duration:
            start_end_secs = [(0.0, duration)]
        elif duration < 2 * clip_duration:
            start_end_secs = [((duration - clip_duration) / 2, (duration + clip_duration) / 2)]
        else:
            num_clip = min(int(duration/clip_duration), max_num_clips)
            start_secs_interval = (duration - clip_duration) / (num_clip - 1)
            start_secs = [start_secs_interval*i for i in range(num_clip)]
            start_end_secs = [(start_sec, start_sec+clip_duration) for start_sec in start_secs]

        # Converting to float otherwise it sometimes yields a fractions.Fraction which is not save-able with json.
        start_end_secs = [(float(a), float(b)) for a, b in start_end_secs]
        clip_timesteps = [np.linspace(start_sec, end_sec, num_frames) for start_sec, end_sec in start_end_secs]
        video_data: List[List["PIL.Image"]] = [video_decoder.decode_frames_at_times(timesteps) for timesteps in clip_timesteps] # (num_clips, [num_frames, PIL.Image])
        video_tensors = [einops.rearrange([torch.from_numpy(np.array(frame)) for frame in frames], 't h w c -> c t h w') for frames in video_data] # (num_clips, [num_frames, c, h, w]

        if ram_cache_path is not None:
            # Saving to RAM cache
            os.makedirs(os.path.dirname(ram_cache_path), exist_ok=True)
            torch.save({'video_tensors': video_tensors, 'start_end_secs': start_end_secs}, ram_cache_path)

        return video_tensors, start_end_secs
    except Exception as e:
        utils.loginfo0(f'Error: {e}')
        return [], []


def transform_video(video_tensor, transforms):
    """Utility to apply preprocessing transformations to a video tensor."""
    video_tensor_pp = transforms(video_tensor)
    video_tensor_pp = video_tensor_pp.permute(1, 0, 2, 3)  # (num_frames, num_channels, height, width)
    return video_tensor_pp

#----------------------------------------------------------------------------

@torch.no_grad()
@hydra.main(config_path="./", config_name="run_camera_clf.yaml", version_base=None)
def run_camera_classifier(cfg):
    # utils.init()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    torch.set_grad_enabled(False)
    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)

    id2label = {'0': 'static', '1': 'dynamic'}

    # Data parameters.
    val_transform = Compose([
        UniformTemporalSubsample(cfg.num_frames),
        Lambda(lambda x: x / 255.0),
        Resize(cfg.resolution),
    ])

    if cfg.files_list_path is not None:
        if cfg.files_list_path.endswith('.txt'):
            with open(cfg.files_list_path, "r") as f:
                video_paths = f.read().splitlines()
                video_paths = [os.path.join(cfg.src_dir, p) for p in video_paths]
        elif cfg.files_list_path.endswith('.csv'):
            import pandas as pd # pylint: disable=import-error
            utils.loginfo0(f'Reading from: {cfg.files_list_path}')
            video_paths = pd.read_csv(cfg.files_list_path)['filepath'].tolist()
            utils.loginfo0(f'Num videos: {len(video_paths)}')
        else:
            raise Exception(f"Unsupported file type: {cfg.files_list_path}")

        # Let's filter out the videos which do not exist
        video_paths = [p for p in video_paths if os.path.isfile(p)]
        utils.loginfo0(f'Num videos after filtering out the non-existing ones: {len(video_paths)}')
    else:
        video_paths = utils.find_videos_in_dir(cfg.src_dir)
    if cfg.shuffle_seed is not None:
        utils.loginfo0(f'Shuffling {len(video_paths)} videos with seed: {cfg.shuffle_seed}')
        np.random.RandomState(cfg.shuffle_seed).shuffle(video_paths)
    if utils.get_world_size() == 1:
        start_idx = cfg.start_idx
        end_idx = cfg.end_idx if cfg.end_idx is not None else len(video_paths)
    else:
        num_videos_to_process = (cfg.end_idx if cfg.end_idx is not None else len(video_paths)) - cfg.start_idx
        num_videos_per_rank = num_videos_to_process // utils.get_world_size()
        start_idx = cfg.start_idx + utils.get_rank() * num_videos_per_rank
        end_idx = start_idx + num_videos_per_rank
    video_paths = video_paths[start_idx:end_idx]
    # logger.info(f'Num videos on rank {utils.get_rank()}: {len(video_paths)}')

    if cfg.skip_existing:
        utils.loginfo0(f'Before filtering existing files: {len(video_paths)}')
        json_files = set(utils.find_files_in_dir(cfg.src_dir, extensions=(".json",)))
        anno_ext = f".camera-clf.json"
        video_paths = [p for p in tqdm(video_paths) if not (os.path.splitext(p)[0] + anno_ext) in json_files]
        utils.loginfo0(f'After filtering existing files: {len(video_paths)}')
    video_paths_batches = [video_paths[i:i+cfg.batch_size] for i in range(0, len(video_paths), cfg.batch_size)] # (num_batches, [batch_size | <batch_size])
    all_results = {}

    video_paths_batches = tqdm(video_paths_batches) if utils.is_main_process() else video_paths_batches
    for current_video_files in video_paths_batches:
        cur_batch_video_filepaths = []
        cur_batch_video_clip_tensors = []
        cur_batch_video_start_end_secs = []

        for video_file in current_video_files:
            # Parse video, each video is split into at most 5 consecutive clips from the start
            video_tensors, start_end_secs = parse_video(video_file, cfg.clip_duration, cfg.num_frames, max_num_clips=cfg.max_num_clips_per_video, ram_cache_dir=cfg.ram_cache_dir)
            for video_tensor, start_end_secs_cur in zip(video_tensors, start_end_secs):
                cur_batch_video_filepaths.append(video_file)
                cur_batch_video_clip_tensors.append(transform_video(video_tensor, val_transform))
                cur_batch_video_start_end_secs.append(start_end_secs_cur)

        if len(cur_batch_video_filepaths) == 0:
            utils.loginfo0('Skipping empty batch')
            continue

        cur_batch_video_clip_tensors = torch.stack(cur_batch_video_clip_tensors).to(device) # [batch_size, num_frames, c, h, w]
        static_camera_probs = torch.as_tensor([detect_camera_type(t, cfg.detection) for t in cur_batch_video_clip_tensors]) # [batch_size]
        camera_probs = torch.stack([static_camera_probs, 1.0 - static_camera_probs], dim=1) # [batch_size, num_classes]

        # collect results
        current_results = defaultdict(list)
        for video_file, probs, start_end_secs in zip(cur_batch_video_filepaths, camera_probs, cur_batch_video_start_end_secs):
            pred_class = int(torch.argmax(probs))
            current_results[video_file].append({
                'pred_class': pred_class,
                'confidence': float(probs[pred_class]),  # 'confidence' is used in the evaluation script
                'probs_all': probs.cpu().numpy().tolist(),
                'start_end_secs': start_end_secs,
            })

        all_results.update(current_results)

        # Also saving per file
        if cfg.save_per_file:
            for video_file, current_result in current_results.items():
                cur_save_path = os.path.splitext(video_file)[0] + anno_ext
                utils.save_json(current_result, cur_save_path, indent=4)

    save_path = f"{cfg.save_path.replace('.json', f'-rank-{utils.get_rank()}.json')}" if cfg.save_path is not None else None
    if save_path is not None:
        # Update majority vote in terms of the numbers of clips for each video file
        majority_vote_results = {}
        vote_results = {}
        for video_file, current_result in all_results.items():
            pred_classes = [res['pred_class'] for res in current_result]
            pred_classes_count = np.bincount(pred_classes)
            majority_vote = np.argmax(pred_classes_count)
            majority_vote_results[video_file] = id2label[str(majority_vote.item())]

            pred_classes_str = [id2label[str(c)] for c in pred_classes]
            vote_results[video_file] = ",".join(pred_classes_str)

        utils.save_json(all_results, save_path, indent=4)
        utils.save_json(majority_vote_results, save_path.replace(".json", ".majority_vote.json"), indent=4)
        utils.save_json(vote_results, save_path.replace(".json", ".vote.json"), indent=4)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_camera_classifier()

#----------------------------------------------------------------------------
