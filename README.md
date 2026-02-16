# Cross-Modality 3D Point Cloud Registration

Systematic evaluation of handcrafted (FPFH) and learning-based (SpinNet) local descriptors for cross-modality 3D point cloud registration on a cultural-heritage dataset.

This repository implements a modular coarse-to-fine registration pipeline and provides experimental scripts for reproducible benchmarking.

---

# 1. Project Structure

```
.
├── environment.yml             # Conda environment specification
│
├── model/
│   ├── Pointnet2_PyTorch/      # PointNet2 backbone (used by SpinNet)
│   └── SpinNet/                # Pretrained SpinNet implementation
│
├── notebook/
│   ├── experiment.ipynb        # Experimental exploration and debugging
│   └── visualization.ipynb     # Visualization and plotting utilities
│
├── result/
│   ├── Combine/                # Results for descriptor concatenation
│   └── FPFH_SpinNet/           # Results for individual descriptors
│
├── script/
│   └── run.py                  # Main entry point for experiments
│
├── src/
│   ├── analysis.py             # Aggregation and statistics computation
│   ├── descriptor.py           # FPFH / SpinNet feature extraction
│   ├── metric.py               # Evaluation metrics (coverage, RMSE, etc.)
│   ├── preprocess.py           # Scaling, voxelization, filtering
│   ├── sweep.py                # Parameter sweep and multi-seed experiments
│   ├── utils.py                # Shared helper functions
│   └── viz.py                  # Visualization utilities (heatmaps, plots)
```

---

# 2. Pipeline Overview

The registration pipeline follows a coarse-to-fine strategy:

1. **Preprocessing**
   - Scale normalization (mm → m)
   - Voxel downsampling
   - Statistical outlier removal
   - Surface normal estimation

2. **Descriptor Extraction**
   - FPFH (handcrafted)
   - SpinNet (learning-based)
   - Combine (descriptor concatenation)

3. **Global Registration**
   - Feature matching
   - RANSAC-based transformation estimation

4. **Fine Registration**
   - ICP refinement

5. **Evaluation**
   - coverage@τ
   - trimmed RMSE
   - qualitative heatmaps
   - runtime profiling

---

# 3. Core Modules (`src/`)

### `preprocess.py`
- Scale alignment (millimeter to meter conversion)
- Voxel-grid downsampling
- Statistical outlier filtering
- Normal estimation

### `descriptor.py`
- FPFH extraction
- SpinNet inference wrapper
- Descriptor concatenation (Combine)

### `metric.py`
- coverage@τ
- trimmed mean
- trimmed RMSE
- Success detection

### `sweep.py`
- Multi-seed experiments
- Sampling ratio (K%) experiments
- Runtime logging

### `analysis.py`
- Aggregation across seeds
- Statistical summary
- CSV export

### `viz.py`
- Distance heatmap visualization
- Runtime trend plotting
- Qualitative comparison figures

---

# 4. Running Experiments


All experiments are launched via `script/run.py`:

The script runs a sweep over multiple keypoint fractions (`--K-fracs`) and random seeds (`--seeds`) for a *single* (ref, mov) pair.

```bash
CKPT="model/SpinNet/pre-trained_models/3DMatch_best.pkl"
DEVICE="cuda:0"
BS=96

python3 -m script.run \
  --ref Data/3D/B4D.obj \
  --mov Data/2D/mov/B4D_10M_Transform.obj \
  --ckpt "$CKPT" \
  --device "$DEVICE" \
  --batch-size "$BS" \
  --save-dir "p2p/B4D" \
  --K-fracs 0.05 0.1 0.2 0.4 \
  --seeds 0 1 2 3 4
```
---

# 5. Results

All experimental outputs are stored under:

```
result/
├── FPFH_SpinNet/
└── Combine/
```

Each run typically stores:

- Estimated transformation matrix
- Runtime breakdown (Feature / RANSAC / ICP)
- Coverage and RMSE metrics
- Success flag

---

# 6. Reproducibility Notes

- SpinNet uses pretrained weights trained on 3DMatch.
- GPU inference is required for SpinNet.
- Default voxel size: `0.005 m`

---

# 7. Dependencies

This project was developed and tested using:

- Python 3.9  
- PyTorch (CUDA-enabled)  
- Open3D  
- NumPy / SciPy  
- scikit-learn  
- Pandas  
- Matplotlib  
- PointNet2 Ops  

## Recommended Setup (Conda)

The exact environment used for the experiments is provided in `environment.yml`.

To recreate the environment:

```bash
conda env create -f environment.yml
conda activate spinnet_cuda12
```

## SpinNet Compatibility

SpinNet is used with official pretrained weights (3DMatch). Pretrained weights can be downloaded from the
[official SpinNet repository](https://github.com/QingyongHu/SpinNet/tree/main/pre-trained_models).
 
The original implementation depends on an older torch version.  
To run inference under the current environment, minor compatibility fixes were applied:

- Updated deprecated PyTorch API calls  
- Adapted PointNet2 CUDA extension build configuration  

These modifications are limited to compatibility and do not alter the model architecture or pretrained weights.


---

# 8. Hardware Configuration

SpinNet and Combine experiments were conducted on:

- NVIDIA GPU V-100 (32GB VRAM)
- Batch size: 96
- Peak memory usage: ~20GB




