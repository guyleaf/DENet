# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DENet (DE-YOLO) is a PyTorch implementation of a Detection-Enhancement Network for object detection under adverse weather conditions (low-light, fog). It is research code from the ACCV 2022 paper. Only evaluation/testing code is provided; training code is not yet released.

The codebase is structured as a pip-installable package (`denet`).

## Setup

```bash
pip install -e .           # core deps only (torch, numpy, opencv) — for extending DENet in other projects
                           # Python >= 3.9
pip install -e ".[full]"   # all deps pinned for paper reproducibility
                           # Python == 3.9
```

`datasets/` and `pretrained_models/` are symlinks to external data directories.

## Running Evaluation

```bash
# ExDark (low-light, 10 classes)
bash scripts/test_exdark_deyolo.sh

# RTTS (foggy, 5 classes)
bash scripts/test_rtts_deyolo.sh

# Or manually:
python scripts/test.py \
  --project yolov3_exdark_10c_deyolo \
  --name deyolo \
  --model deyolo.DEYOLO \
  --img_size_test 544 544 \
  --batch_size 2 \
  --data ./configs/exdark_10c.yaml \
  --hyp ./configs/hyp.voc.scratch.yaml \
  --verbose \
  --nms_thres 0.5 \
  --conf_thres 0.001 \
  --checkpoint pretrained_models/deyolo_lowlight/best.pt
```

TensorBoard logs are written to `./runs/`:
```bash
tensorboard --logdir ./runs
```

There is no test suite or linter configured.

## Architecture

**Execution flow:**
```
scripts/test.py → TestOptions.init() → init_model() → evaluate()
                                                         └→ create_dataloader()
                                                         └→ model.forward() → non_max_suppression() → ap_per_class() / coco_eval()
```

**Model classes:**
- `YOLOv3` (`denet/models/deyolo.py`): Standard YOLOv3 with Darknet53 backbone + FPN neck + Detect head
- `DEYOLO` (`denet/models/deyolo.py`): Extends YOLOv3 — runs the input image through the DENet enhancement module **before** the backbone (not on intermediate features)

**DENet enhancement pipeline** (`denet/core/modules.py`, inside `DEYOLO.forward()`):
1. Laplacian pyramid decomposition splits image into low/high frequency components
2. `Trans_low`: Enhances low-frequency component with multi-scale convolutions and channel/spatial attention
3. `Trans_high`: Enhances high-frequency component using Spatial-Feature Transform (SFT) conditioned on `Trans_low` output
4. Pyramid reconstruction produces the enhanced image fed to the YOLO backbone

**Trans_low compat_mode:** `Trans_low` has a `compat_mode=True` parameter (default). When `True`, it replicates the original code bug (mm1 used for all 4 conv branches). Set `compat_mode=False` for correct multi-scale behavior. Pass through `DENet(compat_mode=...)`.

**Package structure:**
```
denet/
├── __init__.py              # __version__
├── core/
│   ├── __init__.py          # re-exports DENet
│   └── modules.py           # Enhancement module classes (Lap_Pyramid_Conv, Trans_low, DENet, etc.)
├── models/
│   ├── __init__.py          # re-exports YOLOv3, DEYOLO
│   └── deyolo.py            # Detector classes (YOLOv3, DEYOLO, Darknet53, etc.)
├── data/
│   ├── __init__.py          # re-exports create_dataloader
│   ├── datasets.py          # create_dataloader(), yolo_dataset, caching
│   └── augmentations.py     # data augmentation helpers
├── engine/
│   ├── __init__.py          # re-exports init_model, evaluate, options
│   ├── common.py            # init_model, evaluate, get_optimizer, load_checkpoint
│   └── options.py           # BaseOptions, TestOptions, TrainOptions (merged)
└── utils/
    ├── general.py           # misc helpers (file I/O, get_hash)
    ├── torch_utils.py       # model introspection, device setup, import_fun
    ├── yolo_utils.py        # NMS, anchor utilities, coordinate transforms
    ├── metrics.py           # precision-recall, AP, COCO evaluation
    ├── plots.py             # visualization helpers
    ├── ssim.py              # SSIM/MS_SSIM (standalone, no internal deps)
    ├── lr_scheduler.py      # WarmupCosineLR, yolo_lr_scheduler
    ├── feature_vis_tsne.py  # t-SNE feature visualization
    ├── proxy_a_distance.py  # Proxy A-Distance for domain adaptation
    └── pycocotools/         # vendored pycocotools
```

**Key files:**
- `denet/engine/common.py` — `init_model()`, `evaluate()`, `get_optimizer()`, `load_checkpoint()`
- `denet/models/deyolo.py` — all detector classes
- `denet/core/modules.py` — DENet enhancement classes
- `denet/engine/options.py` — argument parsing (BaseOptions, TestOptions, TrainOptions)
- `denet/data/datasets.py` — `create_dataloader()`, `yolo_dataset` (supports `"normal"` and `"train_paired"` modes for supervised enhancement training)
- `denet/utils/yolo_utils.py` — NMS, anchor utilities, coordinate transforms
- `denet/utils/metrics.py` — precision-recall, AP, COCO evaluation
- `configs/*.yaml` — dataset configs and hyperparameters
- `scripts/` — entry point scripts

**Model selection** is done at runtime via `--model deyolo.DEYOLO` or `deyolo.YOLOv3`; `init_model()` uses `importlib` to load from `denet.models.*` dynamically.

**Checkpoint loading** uses a tolerant `_load_state_dict_()` that skips mismatched shapes, allowing partial weight loading.

**Output:** each run creates a timestamped directory under `runs/<task>/<project>/<timestamp>_<name>/`; visualization images (`test_batch*.jpg`) are saved there during evaluation.

## Key Imports

```python
from denet.core import DENet                    # enhancement module
from denet.models import YOLOv3, DEYOLO         # detector models
from denet.engine import init_model, evaluate   # engine functions
from denet.data import create_dataloader        # data loading
from denet.engine import TestOptions            # CLI options
```

## Additional Docs
- Code Plans: @docs/plans
