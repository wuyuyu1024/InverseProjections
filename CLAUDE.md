# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements inverse projection techniques for dimensionality reduction, specifically iLAMP and NNinv methods. The project provides code and demonstrations for inverse multidimensional projections that can map points from low-dimensional spaces back to high-dimensional spaces.

## Core Architecture

The project contains implementations of several inverse projection methods:

### Main Components

- **NNinv.py**: Neural network-based inverse projection using PyTorch
  - `NNinv_net`: Deep neural network with configurable architecture (default: [128, 256, 512, 1024] layers)
  - `NNinv`: Wrapper class with sklearn-compatible API
  - Uses ReLU activations and Sigmoid output, MSE loss, Adam optimizer

- **lamp.py**: Local Affine Multidimensional Projection methods
  - `force_method()`: Forced projection technique for control point placement
  - `lamp2d()`: Main LAMP algorithm implementation
  - `ilamp()`: Inverse LAMP for mapping 2D points back to high-dimensional space
  - `ILAMP`: Sklearn-compatible wrapper for inverse LAMP

- **rbf_inv.py**: Radial Basis Function inverse projection
  - `RBFinv`: RBF interpolation with configurable functions (Gaussian, Multiquadric)
  - Uses scipy for distance calculations and linear system solving

- **multilateration.py**: Multilateration-based inverse projection
  - `MDSinv`: Uses geometric multilateration principles
  - Point selection strategies: nearest, furthest, random

- **gradient_map.py**: Gradient map computation utilities
  - `get_gradient_map()`: Computes gradient maps for inverse projection quality assessment

## Package Structure

The project is organized as a proper Python package using `uv` for dependency management:

```
inverse-projections/
├── src/inverse_projections/         # Main package
│   ├── __init__.py                  # Package exports
│   ├── nninv.py                     # Neural network inverse projection
│   ├── lamp.py                      # LAMP and iLAMP methods
│   ├── rbf.py                       # RBF inverse projection  
│   ├── multilateration.py           # Multilateration-based methods
│   └── gradient_map.py              # Gradient map utilities
├── examples/                        # Example notebooks
│   ├── demo.ipynb                   # Main MNIST demonstration
│   └── demo_gradient_map.ipynb      # Gradient map examples
├── pyproject.toml                   # Package configuration
├── README.md                        # Project documentation
└── CLAUDE.md                        # This file
```

## Development Workflow

### Package Installation
```bash
# Install in development mode with dependencies
uv sync

# Install with examples dependencies
uv sync --extra examples

# Install the package in editable mode
uv pip install -e .
```

### Running Examples
```bash
# Navigate to examples directory
cd examples

# Run main demonstration with MNIST dataset
jupyter notebook demo.ipynb

# Run gradient map demonstration
jupyter notebook demo_gradient_map.ipynb
```

### Package Usage
```python
# Import classes directly from the package
from inverse_projections import NNinv, ILAMP, RBFinv, MDSinv, get_gradient_map

# Use any inverse projection method
nninv = NNinv()
nninv.fit(X2d, X_original)
reconstructed = nninv.transform(new_2d_points)
```

### Dependencies
Package dependencies are managed in `pyproject.toml`:
- PyTorch (torch, torch.nn, torch.optim) 
- scikit-learn (preprocessing, manifold, neighbors, datasets)
- NumPy, SciPy (spatial, linalg)
- Matplotlib, tqdm, pandas
- Optional: Jupyter dependencies for examples

### Code Architecture Patterns

All inverse projection methods follow a consistent sklearn-compatible API:
- `fit(X2d, X)`: Train on 2D projections and original high-dimensional data
- `transform(points_2d)`: Map 2D points to high-dimensional space
- `inverse_transform(points_2d)`: Alias for transform method
- `reset()`: Clear fitted model state

### Key Implementation Details

- **NNinv**: Requires MinMaxScaler normalization, supports GPU acceleration
- **ILAMP**: Uses KDTree for efficient nearest neighbor queries, vectorized operations
- **RBF**: Matrix-based interpolation with batch processing for large datasets
- **Multilateration**: Geometric approach using SVD decomposition and least squares

### Data Processing Notes
- MNIST data is normalized to [0,1] range for neural network training
- 2D projections typically use MDS, t-SNE, or other dimensionality reduction techniques
- Gradient maps use 100x100 grids by default for visualization

### Performance Considerations
- Batch processing is implemented in most methods for memory efficiency
- GPU support available for NNinv when CUDA is available
- Progress bars (tqdm) used for long-running operations