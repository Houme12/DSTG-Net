# DSTG-Net: Dual-Stream Decoupled Spatio-Temporal Graph Network for Traffic Forecasting

![Python](https://img.shields.io/badge/Python-3.8-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12-red)
![License](https://img.shields.io/badge/License-MIT-green)

> Official implementation of **DSTG-Net**, a dual-stream decoupled spatio-temporal graph network for traffic flow forecasting.

---

## 📋 Overview

Traffic flow is inherently composed of two distinct components: a **steady component** driven by periodic patterns (e.g., daily rush hours) and a **dynamic component** triggered by incidents (e.g., accidents). These two components exhibit fundamentally different spatial propagation characteristics, yet existing methods model them with a shared graph structure, forcing a compromise between stability and adaptability.

**DSTG-Net** addresses this by:

1. **Signal Decomposition**: Separates input into steady and dynamic components via a learnable low-pass filter.
2. **Dual-Stream Architecture**:
   - **Steady Stream**: Captures periodic diffusion through a *Time-Slot Memory Bank* and fused static graphs.
   - **Dynamic Stream**: Models incident-driven propagation via *Bilateral Modulation* on a global adaptive skeleton.
3. **Layer-wise Multi-Order Graph Diffusion**: Aligns spatial propagation range with temporal receptive field depth without extra parameters.

##🚀 Usage
python train.py \
    --device cuda:0 \
    --data data/PEMS04-flow \
    --adjdata data/PEMS04-flow \
    --num_nodes 307 \
    --in_dim 3 \
    --seq_length 12 \
    --batch_size 64 \
    --epochs 50 \
    --save ./garage/pems04

##🙏 Acknowledgments
This work builds upon the excellent Graph WaveNet codebase. We thank the authors for making their implementation publicly available.
