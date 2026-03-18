# refactor-to-package.md

## Goal
Refactor this repo into a proper pip-installable Python package. All code should be importable via `from denet.xxx import yyy` — no `sys.path` hacks. The original functionality (training, testing, etc.) must be preserved.

## Context
- This paper (ACCV 2022) proposes a **Laplacian pyramid based** image enhancement module for object detection under adverse conditions
- The core module processes the input image to enhance it **before** it enters any detector backbone
- We need to isolate this image enhancement module from the detector-specific code

## Package Structure

```
DENet/
├── denet/                     # The Python package
│   ├── __init__.py
│   ├── core/                  # The paper's core contribution — minimal deps
│   │   ├── __init__.py        # re-exports public API
│   │   ├── modules.py         # core nn.Module(s) — Laplacian pyramid enhancement
│   │   ├── loss.py            # custom loss (if any, otherwise omit)
│   │   └── utils.py           # utils only used by core (if any)
│   ├── models/                # detector, backbone, etc.
│   ├── data/                  # dataset, dataloader
│   ├── engine/                # trainer, evaluator
│   └── utils/                 # general utilities
├── scripts/ or tools/         # entry-point scripts, import from denet.*
├── configs/                   # experiment configs
├── pyproject.toml
└── CLAUDE.md
```

### Submodule separation
- `denet.core` contains the paper's main contribution: the Laplacian pyramid enhancement module, its loss, and any utils they depend on.
- Other submodules (`denet.models`, `denet.data`, `denet.engine`, `denet.utils`) contain the rest of the codebase.
- **`denet.core` does NOT import from other `denet.*` submodules.** Other submodules CAN import from `denet.core`.

### Core public API
```python
from denet.core import SomeEnhancementModule  # nn.Module
from denet.core import SomeLoss                # if custom loss exists
```

### Core module interface
```python
module = SomeEnhancementModule()
output = module(img)  # img: [B, 3, H, W] → output: [B, 3, H, W]
```
- Input/output: `[B, 3, H, W] → [B, 3, H, W]` (RGB image tensor)
- No hardcoded spatial dimensions (H, W can be anything)
- Pure PyTorch — no detectron2, mmdet, ultralytics, or other framework dependencies
- Differentiable end-to-end (gradients flow through the module)

### Custom loss
If the paper uses a loss function that isn't a standard `torch.nn` loss (MSELoss, L1Loss, etc.), export it in `denet/core/loss.py`. If all losses are standard, skip this and document which loss + weight the paper uses.

## Dependency Management

### `pyproject.toml`
```toml
[project]
name = "denet"
requires-python = ">=3.9"
dependencies = [
    "torch >= 1.13.1",
    # ONLY what denet.core actually imports — keep minimal
]

[project.optional-dependencies]
full = [
    # Everything denet.models, denet.data, denet.engine, scripts/ need
    # For running original training/testing
]
```

```bash
pip install -e .          # core deps only
pip install -e ".[full]"  # everything for original experiments
```

### Version compatibility for `denet.core`
- **PyTorch 1.13.1 through 2.7+** — no `torch.compile` or 2.0+ only APIs
- **NumPy 1.24 through 2.x** — no `np.float`, `np.int` etc. (removed in 2.0)
- Fix incompatible APIs during refactoring; flag behavioral changes for discussion

## Refactoring Steps
1. Read the full codebase, identify:
   - The core enhancement module(s) — the paper's main contribution (Laplacian pyramid based)
   - Any custom loss function(s) that aren't standard PyTorch
   - External framework dependencies to strip from core
   - Other code: detector/backbone → `denet/models/`, data loading → `denet/data/`, training logic → `denet/engine/`, general utils → `denet/utils/`
   - Dependencies: which belong to core vs full
   - API usage incompatible with torch>=1.13.1 or numpy>=1.24,<3
2. Create `denet/` package with submodules
3. Move/extract code into the package structure
4. Update all internal imports to use package paths (`from denet.xxx import yyy`)
5. Update scripts to import from the package
6. Write `pyproject.toml` with split dependencies
7. Fix any version-incompatible API usage in `denet/core/`
8. Smoke test core:
   ```python
   from denet.core import SomeModule
   import torch
   m = SomeModule()
   x = torch.randn(2, 3, 640, 640)
   assert m(x).shape == (2, 3, 640, 640)
   ```
9. Verify original training/testing works with `pip install -e ".[full]"`
