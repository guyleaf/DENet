# Refactor DENet into pip-installable Package

## Context
DENet is a flat research codebase (ACCV 2022) with no package structure — imports work only via CWD. We're refactoring it into a proper pip-installable package (`denet`) so the core enhancement module can be imported as `from denet.core import DENet`. The refactoring plan is defined in `docs/plans/refactor-to-package.md`.

## Design Decisions
- **Trans_low bug (mm1 called 4x):** Add `compat_mode=True` flag — defaults to buggy behavior for checkpoint compat, `False` uses all 4 conv branches correctly
- **cv2 in core:** Keep cv2 as a core dep (`opencv-python>=4.2,<5`), `getGaussianKernel` is stable
- **Old files:** Remove after moving to `denet/` — clean break

## Target Structure
```
DENet/
├── denet/
│   ├── __init__.py           # __version__
│   ├── core/
│   │   ├── __init__.py       # re-exports DENet
│   │   └── modules.py        # Enhancement module classes
│   ├── models/
│   │   ├── __init__.py
│   │   └── deyolo.py         # Detector classes (YOLOv3, DEYOLO, Darknet53, etc.)
│   ├── data/
│   │   ├── __init__.py       # re-exports create_dataloader
│   │   ├── datasets.py
│   │   └── augmentations.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── common.py         # init_model, evaluate, get_optimizer, load_checkpoint
│   │   └── options.py        # BaseOptions, TestOptions, TrainOptions
│   └── utils/
│       ├── __init__.py
│       ├── general.py
│       ├── torch_utils.py
│       ├── yolo_utils.py
│       ├── metrics.py
│       ├── plots.py
│       ├── ssim.py           # SSIM/MS_SSIM (evaluation metric, standalone)
│       ├── lr_scheduler.py
│       ├── feature_vis_tsne.py
│       ├── proxy_a_distance.py
│       └── pycocotools/      # vendored, copy as-is
├── scripts/
│   ├── test.py
│   ├── test_exdark_deyolo.sh
│   └── test_rtts_deyolo.sh
├── configs/
│   ├── exdark_10c.yaml
│   ├── rtts_5c.yaml
│   └── hyp.voc.scratch.yaml
├── pyproject.toml
└── CLAUDE.md
```

---

## Tasks

### Task 1: Create package skeleton + pyproject.toml
**Scope:** Create directory tree, all `__init__.py` files, `pyproject.toml`.

**Create:**
- `denet/__init__.py` — `__version__ = "0.1.0"`
- `denet/core/__init__.py` — empty (populated in Task 2)
- `denet/models/__init__.py` — empty
- `denet/data/__init__.py` — empty
- `denet/engine/__init__.py` — empty
- `denet/utils/__init__.py` — empty
- `scripts/` — empty dir (create `.gitkeep`)
- `configs/` — empty dir (create `.gitkeep`)
- `pyproject.toml`:
  ```toml
  [build-system]
  requires = ["setuptools>=64", "wheel"]
  build-backend = "setuptools.build_meta"

  [project]
  name = "denet"
  version = "0.1.0"
  description = "DENet: Detection-Enhancement Network for object detection under adverse weather"
  requires-python = ">=3.9"
  dependencies = [
      "torch>=1.13.1,<3",
      "numpy>=1.24,<3",
      "opencv-python>=4.2,<5",
  ]

  [project.optional-dependencies]
  full = [
      "torchvision>=0.14,<1",
      "tensorboard>=2.5,<3",
      "PyYAML>=5.1,<7",
      "pytz",
      "scikit-learn>=0.21,<2",
      "scipy>=1.5,<2",
      "thop<1",
      "tqdm<5",
      "matplotlib>=3.1,<4",
  ]

  [tool.setuptools.packages.find]
  include = ["denet*"]
  ```

**Verify:** `pip install -e . && python -c "import denet; print(denet.__version__)"`

---

### Task 2: Extract core enhancement modules → `denet/core/modules.py`
**Scope:** Extract DENet enhancement classes from `models/deyolo.py` lines 253-628.

**Create:** `denet/core/modules.py`

**Classes to extract (in order):**
1. `Lap_Pyramid_Conv` (L253-306) — keep cv2 usage as-is
2. `ChannelAttention` (L323-337)
3. `SpatialAttention` (L340-357)
4. `Trans_guide` (L360-372)
5. `Trans_low` (L375-424) — add `compat_mode` parameter (see below)
6. `SFT_layer` (L427-447)
7. `Trans_high` (L450-457)
8. `Up_guide` (L460-474)
9. `DENet` (L589-628) — pass `compat_mode` through to `Trans_low`

**Skip:** `ResidualBlock` (L309-320) — unused, don't extract

**Imports:**
```python
import cv2
import numpy as np
import torch
import torch.nn as nn
```
No imports from other `denet.*` submodules.

**Trans_low compat_mode implementation:**
```python
class Trans_low(nn.Module):
    def __init__(self, ch_blocks=64, ch_mask=32, compat_mode=True):
        ...
        self.compat_mode = compat_mode
        ...

    def forward(self, x):
        ...
        if self.compat_mode:
            # Original code bug: mm1 used for all 4 branches
            x1_1 = self.mm1(x1)
            x1_2 = self.mm1(x1)
            x1_3 = self.mm1(x1)
            x1_4 = self.mm1(x1)
        else:
            x1_1 = self.mm1(x1)
            x1_2 = self.mm2(x1)
            x1_3 = self.mm3(x1)
            x1_4 = self.mm4(x1)
        ...
```

**DENet constructor:** Add `compat_mode=True` param, pass to `Trans_low`.

**Update `denet/core/__init__.py`:**
```python
from denet.core.modules import DENet
```

**Verify:**
```python
import torch
from denet.core import DENet
m = DENet()
x = torch.randn(2, 3, 64, 64)
assert m(x).shape == (2, 3, 64, 64)
```

---

### Task 3: Move leaf utilities → `denet/utils/`
**Scope:** Move utility modules with no internal imports.

**Copy as-is (no changes needed):**
| Source | Destination |
|--------|-------------|
| `utils/general.py` | `denet/utils/general.py` |
| `utils/ssim.py` | `denet/utils/ssim.py` |
| `utils/lr_scheduler.py` | `denet/utils/lr_scheduler.py` |
| `utils/feature_vis_tsne.py` | `denet/utils/feature_vis_tsne.py` |
| `utils/proxy_a_distance.py` | `denet/utils/proxy_a_distance.py` |
| `utils/pycocotools/` (all files) | `denet/utils/pycocotools/` |

**Copy as-is (keep module-level side effects like `cv2.setNumThreads(0)`):**
| Source | Destination |
|--------|-------------|
| `utils/torch_utils.py` | `denet/utils/torch_utils.py` |

**Verify:**
```python
from denet.utils.torch_utils import is_parallel, initialize_weights
from denet.utils.general import get_hash
```

---

### Task 4: Move utils with internal deps → `denet/utils/`
**Depends on:** Task 3

**Copy with import updates:**

| Source | Dest | Import changes |
|--------|------|---------------|
| `utils/yolo_utils.py` | `denet/utils/yolo_utils.py` | `utils.torch_utils` → `denet.utils.torch_utils` |
| `utils/plots.py` | `denet/utils/plots.py` | `utils.yolo_utils` → `denet.utils.yolo_utils` |
| `utils/metrics.py` | `denet/utils/metrics.py` | `utils.plots` → `denet.utils.plots`; `utils.yolo_utils` → `denet.utils.yolo_utils`; `utils.pycocotools` → `denet.utils.pycocotools` |

**Verify:**
```python
from denet.utils.yolo_utils import non_max_suppression
from denet.utils.metrics import ap_per_class
```

---

### Task 5: Move data loading → `denet/data/`
**Depends on:** Task 4

**Copy with import updates:**

| Source | Dest | Import changes |
|--------|------|---------------|
| `utils/augmentations.py` | `denet/data/augmentations.py` | `utils.yolo_utils` → `denet.utils.yolo_utils` |
| `utils/datasets.py` | `denet/data/datasets.py` | `utils.augmentations` → `denet.data.augmentations`; `utils.general` → `denet.utils.general`; `utils.yolo_utils` → `denet.utils.yolo_utils` |

**Compatibility fix:** `torch.load()` calls — version-aware wrapper:
```python
def _torch_load(f, map_location=None):
    """torch.load with weights_only=False for PyTorch >= 1.13, plain call for older."""
    try:
        return torch.load(f, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(f, map_location=map_location)
```

**Update `denet/data/__init__.py`:**
```python
from denet.data.datasets import create_dataloader
```

---

### Task 6: Move models → `denet/models/`
**Depends on:** Tasks 2, 4

**Create:** `denet/models/deyolo.py`

**Extract from `models/deyolo.py`** — detector classes only:
- `CBL`, `Resblock`, `DownSample`, `ResblockX` (L12-71)
- `Darknet53` (L74-110)
- `Upsample`, `CBLX5`, `Neck` (L113-174)
- `YOLO_Head`, `Detect` (L177-250)
- `YOLO_BASE` (L478-564)
- `YOLOv3` (L567-586)
- `DEYOLO` (L631-674)

**Import changes:**
```python
from denet.utils.torch_utils import initialize_weights, is_parallel, model_info
from denet.utils.yolo_utils import check_anchor_order
from denet.core.modules import DENet  # for DEYOLO
```
Remove `import cv2`, `import numpy as np` (not needed by detector code).

**Compatibility fix:** `torch.meshgrid` in `Detect._make_grid` — version-aware call:
```python
try:
    yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing='ij')
except TypeError:
    yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx))
```

**Update `denet/models/__init__.py`:**
```python
from denet.models.deyolo import YOLOv3, DEYOLO
```

---

### Task 7: Move engine code → `denet/engine/`
**Depends on:** Tasks 4, 5, 6

**Create:** `denet/engine/options.py`
- Merge `options/base_options.py`, `test_options.py`, `train_options.py` into one file
- Replace relative imports with direct class references (all in same file)

**Create:** `denet/engine/common.py`
- Source: `common.py`
- Import changes: all `utils.*` → `denet.utils.*`
- In `init_model()`: change `import_fun("models", ...)` to `import_fun("denet.models", ...)`
- `torch.load()` calls: use version-aware `_torch_load()` wrapper (same as Task 5)

**Update `denet/engine/__init__.py`:**
```python
from denet.engine.common import init_model, evaluate, get_optimizer, load_checkpoint
from denet.engine.options import BaseOptions, TestOptions, TrainOptions
```

---

### Task 8: Create scripts + move configs
**Depends on:** Task 7

**Create:** `scripts/test.py`
- Based on `test.py`, all imports from `denet.*`
- `torch.load()` calls: use version-aware `_torch_load()` wrapper (same as Task 5)

**Create:** `scripts/test_exdark_deyolo.sh`, `scripts/test_rtts_deyolo.sh`
- Update `python test.py` → `python scripts/test.py`
- Update config paths: `./data/*.yaml` → `./configs/*.yaml`, `./hyp/*.yaml` → `./configs/*.yaml`

**Move configs:**
- `data/exdark_10c.yaml` → `configs/exdark_10c.yaml`
- `data/rtts_5c.yaml` → `configs/rtts_5c.yaml`
- `hyp/hyp.voc.scratch.yaml` → `configs/hyp.voc.scratch.yaml`

**Verify:** `python scripts/test.py --help`

---

### Task 9: Delete old files + update docs + final verify
**Depends on:** All previous tasks

**Delete old files:**
- `common.py`
- `test.py`
- `test_exdark_deyolo.sh`, `test_rtts_deyolo.sh`
- `models/deyolo.py`, `models/__pycache__/`
- `options/` (entire directory)
- `utils/` (entire directory)
- `data/` (directory with yaml files — NOT the `datasets/` symlink)
- `hyp/` (entire directory)
- `requirements.txt` (replaced by pyproject.toml)

**Update `CLAUDE.md`:** Reflect new package structure, new commands, new import paths.

Note: `.gitignore` already has `*.egg-info/`, `dist/`, `build/` — no changes needed.

**Final verification:**
```bash
pip install -e .
python -c "from denet.core import DENet; import torch; m=DENet(); assert m(torch.randn(1,3,128,128)).shape==(1,3,128,128); print('core OK')"
pip install -e ".[full]"
python -c "from denet.engine import init_model; print('engine OK')"
python scripts/test.py --help
```

---

## Task Dependency Graph
```
Task 1 (skeleton)
├── Task 2 (core/modules) ─────────┐
├── Task 3 (leaf utils)             │
│   └── Task 4 (utils with deps)   │
│       ├── Task 5 (data/)         │
│       └── Task 6 (models/) ←─────┘
│           └── Task 7 (engine/)
│               └── Task 8 (scripts + configs)
│                       └── Task 9 (cleanup + verify)
```

**Parallelizable:** Tasks 2 and 3 can run in parallel after Task 1. Tasks 5 and 6 can run in parallel after Task 4.

## Key Files Reference
- `models/deyolo.py` — split between `denet/core/modules.py` (L253-628) and `denet/models/deyolo.py` (L12-250, L478-674)
- `common.py` → `denet/engine/common.py`
- `options/*.py` → `denet/engine/options.py` (merged)
- `utils/ssim.py` → `denet/utils/ssim.py`
- `utils/torch_utils.py` → `denet/utils/torch_utils.py`
- `utils/datasets.py` → `denet/data/datasets.py`
- `utils/augmentations.py` → `denet/data/augmentations.py`
