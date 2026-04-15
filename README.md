# FedWSOComp: Communication-Efficient Federated Learning for Brain Tumour Segmentation

## 📌 Overview
This repository implements a federated learning framework for 3D medical image segmentation, focusing on communication efficiency through sparsification and weight clustering.

The method is designed for distributed medical datasets (MRI/CT) and aims to reduce communication overhead while maintaining high segmentation performance.

---

## 🧠 Key Features

- Federated Learning using Flower framework
- Weight sharing optimization (FedWSOComp)
- Sparsification (Top-k / magnitude pruning)
- Quantization via clustering
- Communication-efficient training
- 3D brain tumour segmentation (FeTS/BraTS)

---

## 📂 Project Structure
FeTSxFedWSO/
├── client_app.py
├── server_app.py
├── FedWSOCompStrategy.py
├── train.py
├── utils.py
├── brats.py
├── data_analysis.ipynb
│
├── clients/
│ └── BrainTumorSegmentation3dClient/
│ ├── ClientImpl.py
│ ├── loading_utils.py
│ ├── utils.py
│
├── server_visualize.py
├── Resultsplots.py
├── test.p
