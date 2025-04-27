# Script for Metric Scaling of Camera Paths in a video


## Install

```
pip install -r requirements.txt
source install_colmap.sh 1 # use 1 with CUDA support, 0 for CPU-only
```


## Run Metric Scaling with known cameras

```
python3 video_cam_metric_rescale.py -v demo/018f7907401f2fef/018f7907401f2fef.mp4 -c demo/018f7907401f2fef/018f7907401f2fef.camera_info.json
```

## Run Metric Scaling with unknown cameras

```
python3 video_cam_metric_rescale.py -v demo/018f7907401f2fef/018f7907401f2fef.mp4
```
