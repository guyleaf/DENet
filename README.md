# DE-YOLO

PyTorch implementation of **DENet: Detection-driven Enhancement Network for Object Detection under Adverse Weather Conditions** (ACCV 2022).

## Notice
It is refactored version. So, the evaluation results may slightly differ from original paper. I only confirm the results by test scripts.

![DE-YOLO architecture](figs/DE-YOLO.jpg)

## Installation

```bash
# Core deps only (torch, opencv) — for extending DENet in other projects
pip install -e .

# Full deps — for reproducing paper results (requires Python == 3.9)
pip install --extra-index-url https://download.pytorch.org/whl/cu117 -e ".[full]"
```

**Core dependencies:**
- `torch>=1.13.1,<3`
- `opencv-python>=4.2,<5`

**Full dependencies** (pinned for paper reproducibility):
- `torch==1.13.1`
- `torchvision==0.14.1`
- `numpy<2`
- `tensorboard>=2.5,<3`
- `PyYAML>=5.1,<6`
- `scikit-learn>=0.21,<1`
- `scipy>=1.5,<2`
- `thop<1`
- `tqdm<5`
- `matplotlib>=3.1,<4`

`datasets/` and `pretrained_models/` should be symlinks (or directories) pointing to the respective data.

## Datasets and Models

Download the processed datasets and pretrained models:

- [ExDark](https://github.com/NIvykk/research_demo/releases/download/V1.0/ExDark.zip) — low-light, 10 classes
- [RTTS](https://github.com/NIvykk/research_demo/releases/download/V1.0/RTTS.zip) — natural fog, 5 classes
- [Pretrained Models](https://github.com/NIvykk/research_demo/releases/download/V1.0/pretrained_models.zip)

Expected layout after extraction:

```
DENet/
├── datasets/
│   ├── ExDark/
│   └── RTTS/
└── pretrained_models/
    ├── deyolo_lowlight/
    │   └── best.pt
    └── deyolo_foggy/
        └── best.pt
```

## Evaluation

```bash
# ExDark (low-light, 10 classes)
bash scripts/test_exdark_deyolo.sh

# RTTS (foggy, 5 classes)
bash scripts/test_rtts_deyolo.sh
```

Results and visualizations are saved under `runs/`. TensorBoard logs can be viewed with:

```bash
tensorboard --logdir ./runs
```

## Training

Training code is not released with this repository.

## Package Structure

```
denet/
├── core/modules.py      # DENet enhancement module (Laplacian pyramid, Trans_low, Trans_high)
├── models/deyolo.py     # Detector classes (YOLOv3, DEYOLO, Darknet53, Neck, Detect)
├── engine/
│   ├── common.py        # init_model(), evaluate(), get_optimizer(), load_checkpoint()
│   └── options.py       # BaseOptions, TestOptions, TrainOptions
├── data/
│   ├── datasets.py      # create_dataloader(), yolo_dataset
│   └── augmentations.py
└── utils/               # NMS, metrics, plots, SSIM, LR scheduler, pycocotools
```
