# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DENet (DE-YOLO) is a PyTorch implementation of a Detection-Enhancement Network for object detection under adverse weather conditions (low-light, fog). It is research code from the ACCV 2022 paper. Only evaluation/testing code is provided; training code is not yet released.

## Setup

```bash
pip install -r requirements.txt
# Python 3.7.5, PyTorch 1.7.1
```

`datasets/` and `pretrained_models/` are symlinks to external data directories.

## Running Evaluation

```bash
# ExDark (low-light, 10 classes)
bash test_exdark_deyolo.sh

# RTTS (foggy, 5 classes)
bash test_rtts_deyolo.sh

# Or manually:
python test.py \
  --project yolov3_exdark_10c_deyolo \
  --name deyolo \
  --model deyolo.DEYOLO \
  --img_size_test 544 544 \
  --batch_size 2 \
  --data ./data/exdark_10c.yaml \
  --hyp ./hyp/hyp.voc.scratch.yaml \
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
test.py → TestOptions.init() → init_model() → evaluate()
                                                 └→ create_dataloader()
                                                 └→ model.forward() → non_max_suppression() → ap_per_class() / coco_eval()
```

**Model classes** (both in `models/deyolo.py`):
- `YOLOv3`: Standard YOLOv3 with Darknet53 backbone + FPN neck + Detect head
- `DEYOLO`: Extends YOLOv3 — runs the input image through the DENet enhancement module **before** the backbone (not on intermediate features)

**DENet enhancement pipeline** (inside `DEYOLO.forward()`):
1. Laplacian pyramid decomposition splits image into low/high frequency components
2. `Trans_low`: Enhances low-frequency component with multi-scale convolutions and channel/spatial attention
3. `Trans_high`: Enhances high-frequency component using Spatial-Feature Transform (SFT) conditioned on `Trans_low` output
4. Pyramid reconstruction produces the enhanced image fed to the YOLO backbone

**Key files:**
- `common.py` — `init_model()`, `evaluate()`, `get_optimizer()`, `load_checkpoint()`
- `models/deyolo.py` — all model classes
- `options/` — argument parsing; `BaseOptions` handles device, seed, TensorBoard, and output dirs; `TestOptions`/`TrainOptions` add task-specific args
- `utils/datasets.py` — `create_dataloader()`, dataset classes, caching
- `utils/yolo_utils.py` — NMS, anchor utilities, coordinate transforms
- `utils/metrics.py` — precision-recall, AP, COCO evaluation
- `data/*.yaml` — dataset configs (paths, class names)
- `hyp/hyp.voc.scratch.yaml` — hyperparameters (loss weights, augmentation, optimizer)

**Model selection** is done at runtime via `--model deyolo.DEYOLO` or `deyolo.YOLOv3`; `init_model()` uses `importlib` to load the class dynamically.

**Checkpoint loading** uses a tolerant `_load_state_dict_()` that skips mismatched shapes, allowing partial weight loading.

**Output:** each run creates a timestamped directory under `runs/<project>/<name>/`; visualization images (`test_batch*.jpg`) are saved there during evaluation.
