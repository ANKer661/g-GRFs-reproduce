# General Graph Random Features (g-GRFs): Reproduction and Optimization

This repository provides an optimized implementation of the core experiments from the ICLR 2024 paper "General Graph Random Features" by Reid et al.

- Paper: https://arxiv.org/abs/2310.04859
- Original Code: https://github.com/isaac-reid/general_graph_random_features

## Project Overview

The official implementation provided by the authors faced performance limitations and included only a single experiment, making it difficult to fully reproduce the results presented in the paper. This repository addresses these challenges with the following improvements:

1. **Full Reproduction**: We reproduce the main experiments discussed in the paper and generate the corresponding figures.

2. **Performance Optimization**: The algorithm implementation has been optimized to achieve over 10x speedup (for the first experiment) compared to the original codebase. 

3. **File Structure**:
   - `g-GRF.ipynb`: A Jupyter Notebook containing the core algorithm logic and plotting code.

   - `utils.py`: Contains functions for graph processing and auxiliary calculations.

## Installation and Usage

### 1. Environment Setup

We recommend creating a Conda environment using Python 3.12:

```bash
conda env create -f environment.yml
conda activate g-grfs

pip3 install torch torchvision torchaudio  # we only need cpu version
pip install trimesh  # used to load 3D meshes
```

### 2. Running the Code
Open the `g-GRF.ipynb` notebook in Jupyter and run the cells to reproduce the experiments and generate the figures.