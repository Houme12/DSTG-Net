# DSTG-Net: Dual-Stream Decoupled Spatio-Temporal Graph Network for Traffic Forecasting

![Python](https://img.shields.io/badge/Python-3.8-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12-red)
![License](https://img.shields.io/badge/License-MIT-green)

> Official implementation of **DSTG-Net**, a dual-stream decoupled spatio-temporal graph network for traffic flow forecasting. Hope this code is helpful for your research!

---

## 📋 Overview

Traffic flow is inherently composed of two distinct components: a **steady component** driven by periodic patterns (e.g., daily rush hours) and a **dynamic component** triggered by incidents (e.g., accidents). These two components exhibit fundamentally different spatial propagation characteristics, yet existing methods model them with a shared graph structure, forcing a compromise between stability and adaptability.

**DSTG-Net** addresses this by:

1. **Signal Decomposition**: Separates input into steady and dynamic components via a learnable low-pass filter.
2. **Dual-Stream Architecture**:
   - **Steady Stream**: Captures periodic diffusion through a *Time-Slot Memory Bank* and fused static graphs.
   - **Dynamic Stream**: Models incident-driven propagation via *Bilateral Modulation* on a global adaptive skeleton.
3. **Layer-wise Multi-Order Graph Diffusion**: Aligns spatial propagation range with temporal receptive field depth without extra parameters.

---

## 🚀 Usage

### Train

```bash
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
---

## 🙏 Acknowledgments

This work is built upon the [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet) framework. We sincerely thank the authors for their excellent work and open-source contribution.
