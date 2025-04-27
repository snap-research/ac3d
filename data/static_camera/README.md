## Installation
Install the environment with the following command:
```bash
conda env create --prefix ./env -f environment.yml
conda activate ./env
```

## How it works
At the high level, it takes the frame borders of a video, and then checks how close they are by checking if the majority of the pixels are close enough to each other (up to some tolerance since they could differ to due video encoding errors).
Besides, we also consider the video to be dynamic if we are very uncertain about the prediction, which happens when the colors are quite uniform in the borders (e.g., happens for sky or uniform walls).
We check for this by doing the kNN clustering with 3 clusters and checking the average distance of each pixel to its cluster center.

## Run
Here is an example run command (we would suggest trying our original parameters first):
```
torchrun --nproc-per-node=auto --standalone --max-restarts=0 run_camera_clf.py src_dir=/path/to/mp4/videos/ skip_existing=false shuffle_seed=42 save_path=/where/to/save/results
```
The options are in `configs/run_camera_clf.yaml`:
```yaml
# A directory with video files to process. The script would try to find all video files in this directory.
src_dir: ~

# Batch size during inference.
batch_size: 16

# In case we want to process only a subset of the files,
# specify the start/end indicies (useful for multi-node parallel runs).
start_idx: 0
end_idx: ~

# Where to save the results.
save_path: ~

# Should we save the results per file or just dump everything into one file at the end?
# Saving per file is useful for debugging or some data pipelines.
save_per_file: false

num_frames: 15 # Number of frames to extract from each video.
resolution: [256, 256] # To which size should we resize the frames.
clip_duration: 5.0 # Duration of each clip in seconds.

# Sometimes, searching for all video files in a directory is expensive (for non-local storages). You can input a pre-computed list of files.
files_list_path: ~

# Should we ignore the files that have already been processed? The detection is based on the stored annotations run with save_per_file=true.
skip_existing: true

# How many clips to extract from each video.
max_num_clips_per_video: 16
shuffle_seed: ~

detection:
  border_size: 13
  tol: 0.020010004748436956
  static_pixels_ratio_thresh: 0.8442560162008423 # If the colors std is below this threshold, we consider the frame to be moving due to uncertainty.
  min_cluster_dist: 0.005536400073834054
  num_clusters: 2
```

If save_per_file=false, then the results will be saved per rank with the corresponding rank number appended to the `save_path` argument.

By default, we set the hyperparameters for detection for maximum precsion of 0.95 and minimal recall of 0.1.
For some applications, it might be more beneficial to have a higher recall. We ran some hyperparameter search (on an internal manually annotated dataset of 3457 videos with 689 positive samples) and got the following results:
- precision=0.610 & recall=0.420 => `detection: {tol: 0.02843497224770407, static_pixels_ratio_thresh: 0.645595804070981, border_size: 6, min_cluster_dist: 0.0029739421700540527, num_clusters: 1}`
- precision=0.720 & recall=0.327 => `detection: {tol: 0.02201000855175674, static_pixels_ratio_thresh: 0.6672301022616912, border_size: 4, min_cluster_dist: 0.0011242866705639875, num_clusters: 1}`
- precision=0.814 & recall=0.230 => `detection: {tol: 0.019551690681217305, static_pixels_ratio_thresh: 0.7240808950429893, border_size: 6, min_cluster_dist: 0.0013426280724656205, num_clusters: 1}`
- precision=0.920 & recall=0.121 => `detection: {tol: 0.017118190537933998, static_pixels_ratio_thresh: 0.775245528398448, border_size: 8, min_cluster_dist: 0.011314504456316633, num_clusters: 6}`
- precision=0.957 & recall=0.100 => `detection: {tol: 0.020010004748436956, static_pixels_ratio_thresh: 0.8442560162008423, border_size: 13, min_cluster_dist: 0.005536400073834054, num_clusters: 2}`

You can also increase the precision by decreasing the FPS (i.e. increasing `clip_duration` or decreasing `num_frames`).

## Evaluation
If you have a GT csv file with the labels, you can evaluate the results with the following command:
```
python eval.py <results_dir> <gt_csv> --save_path <where_to_save_preds.csv>
```
