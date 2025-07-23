# The Effectiveness of Machine Learning Models for Newtonian Physics Simulations

**Author:** Luan A. Tobias  
**Affiliation:** Concordia International School Hanoi / Polygence  
**Contact:** luan.tobias@concordiahanoi.org

## 🧠 Overview

This repository contains code, data, and results for the research paper:
> *The Effectiveness of Machine Learning Models for Newtonian Physics Simulations*

The goal: Evaluate whether machine learning (ML), specifically neural networks, can simulate Newtonian physics systems (projectile motion and linear drag projectile motion) more efficiently than traditional methods.

## 📂 Project Structure

- **data/**: Simulation datasets (kinematics, LDPM)
- **notebooks/**: Training, analysis, and experiment notebooks (Colab/Jupyter)
- **src/**: Core simulation and ML model code
- **results/**: Accuracy curves, benchmarks, and generated figures
- **paper/**: Drafts and materials for the research paper

## ⚡ Key Features

- Simulation code for classical projectile motion and LDPM systems in Python/NumPy
- On-the-fly data generation for robust ML training and reduced overfitting
- PyTorch models with various architectures, trained and benchmarked
- Scripts for comparing inference time and accuracy versus analytical solutions
- Metrics implemented: RMSE, MAE, and a custom accuracy formula
- All experiments and findings documented in notebooks and paper drafts

## 🚀 Quick Start

1. Clone the repository:
    ```bash
    git clone https://github.com/Lu4nAlmeida/cnm-simnet.git
    cd cnm-simnet
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run simulations or ML training via notebooks in the `notebooks/` folder.

## 📊 Results

See the `results/` folder for:
- Accuracy plots
- Inference benchmarks
- Figures for the paper

## 💡 Research Highlights

- Models trained on infinite synthetic data for robust generalization
- ML model performance compared to analytical CNM solutions
- Metrics and benchmarking scripts included

## 📄 Paper Draft

Find the latest draft and references in the `paper/` folder.

## 🤝 Contributing

Interested in helping or using this work? Feel free to reach out or open an issue!

---

> “Code is like humor. When you have to explain it, it’s bad.” – Cory House

---

## 📬 Contact

- Email: luan.tobias@concordiahanoi.org