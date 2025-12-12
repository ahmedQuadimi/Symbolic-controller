<div align="center">

# 🎮 Symbolic Controller

**A framework for Symbolic Control in 2D, 3D, and Physics-based Environments.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)](https://github.com/ahmedQuadimi/Symbolic-controller)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## 📖 About The Project

This repository implements **Symbolic Control** algorithms designed to synthesize controllers for dynamic systems. It bridges the gap between high-level logic and low-level control, featuring:
* **2D & 3D Abstractions:** Visualizing automata and state-space partitions.
* **Physics Simulation:** Integration with **PyBullet** for realistic dynamic testing.
* **Interactive Development:** Scripts optimized for VS Code interactive cells.

---

## ⚡ Quick Start

### 1. Prerequisites
Ensure you have Python installed. The project relies on specific dependencies found in `requirements.txt`.

### 2. Installation

Clone the repo and install the required packages:

```bash
# Clone the repository
git clone [https://github.com/ahmedQuadimi/Symbolic-controller](https://github.com/ahmedQuadimi/Symbolic-controller)

# Navigate to the directory
cd Symbolic-controller

# Install dependencies
python -m pip install -r requirements.txt


Component,Command,Description
3D Simulation,python 3DTotomata.py,Runs the 3D symbolic automata environment.
2D Simulation,python 2DTotomata.py,Runs the 2D symbolic automata environment.
Physics Sim,python mybullet.py,Launches the PyBullet physics engine simulation.

Symbolic-controller/
├── 2DTotomata.py       # 2D Controller Implementation
├── 3DTotomata.py       # 3D Controller Implementation
├── mybullet.py         # PyBullet Physics Environment
├── requirements.txt    # Python Dependencies
└── README.md           # Documentation