# AC3D: Analyzing and Improving 3D Camera Control in Video Diffusion Transformers

![Teaser](./assets/teaser.png)

| [Project Page](https://snap-research.github.io/ac3d/) | [Paper](https://arxiv.org/abs/2411.18673) |
 
### Information

This is a version of AC3D built on [CogVideoX](https://github.com/THUDM/CogVideo/tree/main). AC3D is a camera-controlled video generation pipeline that follows the plucker-conditioned ControlNet architecture originally introduced in [VD3D](https://snap-research.github.io/vd3d/).

### Installation

Install PyTorch first (we used PyTorch 2.4.0 with CUDA 12.4).

```bash
pip install -r requirements.txt
```

### Dataset

Prepare the [RealEstate10K](https://google.github.io/realestate10k/download.html) dataset following the instructions in [CameraCtrl](https://github.com/hehao13/CameraCtrl). The dataset path will be used for video_root_dir in the train and inference scripts. This is the folder structure after pre-processing:

```
- RealEstate10k
  - annotations
    - test.json
    - train.json
  - pose_files
    - 0000cc6d8b108390.txt
    - 00028da87cc5a4c4.txt
    - ...
  - video_clips
    - 0000cc6d8b108390.mp4
    - 00028da87cc5a4c4.mp4
    - ...
```

### Pre-trained ControlNet models

AC3D: CogVideoX-2B: [Checkpoint](https://drive.google.com/file/d/1RmTnF7mJ65s5TSqr4k_cthZXMWesd3nA/view)

AC3D: CogVideoX-5B: [Checkpoint](https://drive.google.com/file/d/1QsfmLmb-_Pv_pSbLrmbqBBehc9Oo6A79/view)

### Inference scripts

AC3D: CogVideoX-2B
```bash
bash scripts/inference_2b.sh
```

AC3D: CogVideoX-5B
```bash
bash scripts/inference_5b.sh
```

### Training requirements

The 2B model requires 48 GB memory and the 5B model requires 80 GB memory. Using one node with 8xA100 80 GB should take around 1-2 days for the model to converge.

### Training scripts

AC3D: CogVideoX-2B
```bash
bash scripts/train_2b.sh
```

AC3D: CogVideoX-5B
```bash
bash scripts/train_5b.sh
```

### Acknowledgements

- This code mainly builds upon [CogVideoX-ControlNet](https://github.com/TheDenk/cogvideox-controlnet)
- This code uses the original CogVideoX model [CogVideoX](https://github.com/THUDM/CogVideo/tree/main)
- The data procesing and data loading pipeline builds upon [CameraCtrl](https://github.com/hehao13/CameraCtrl)

### Cite

```
@article{bahmani2024ac3d,
  author = {Bahmani, Sherwin and Skorokhodov, Ivan and Qian, Guocheng and Siarohin, Aliaksandr and Menapace, Willi and Tagliasacchi, Andrea and Lindell, David B. and Tulyakov, Sergey},
  title = {AC3D: Analyzing and Improving 3D Camera Control in Video Diffusion Transformers},
  journal = {arXiv preprint arXiv:2411.18673},
  year = {2024},
}
```
