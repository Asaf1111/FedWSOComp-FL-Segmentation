# FedWSOComp: Communication-Efficient Federated Learning for Brain Tumour Segmentation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Flower-orange)
![Task](https://img.shields.io/badge/Task-3D%20Segmentation-green)
![Dataset](https://img.shields.io/badge/Dataset-FeTS%20%2F%20BraTS-purple)

## Overview

FedWSOComp is a federated learning framework for communication-efficient 3D medical image segmentation. It is designed for distributed MRI datasets and reduces communication overhead during training through sparsification and weight clustering, while maintaining high segmentation performance.

The framework targets the [FeTS](https://fets-ai.github.io/Front-End/) / [BraTS](https://www.synapse.org/#!Synapse:syn51156910/wiki/) challenge and is built on top of the [Flower](https://flower.dev/) federated learning framework.

---

## Key Features

- **Federated Learning** via the Flower framework
- **FedWSOComp strategy** — custom aggregation with weight sharing optimization
- **Sparsification** — Top-k and magnitude-based pruning
- **Quantization** — weight clustering for compact model updates
- **Communication efficiency** — significantly reduced bandwidth usage per round
- **3D brain tumour segmentation** on FeTS / BraTS datasets

---

## Project Structure

```
FeTSxFedWSO/
├── client_app.py               # Flower client entry point
├── server_app.py               # Flower server entry point
├── FedWSOCompStrategy.py       # Custom federated strategy (FedWSOComp)
├── train.py                    # Training loop
├── utils.py                    # General utilities
├── brats.py                    # BraTS dataset handling
├── data_analysis.ipynb         # Exploratory data analysis notebook
│
├── clients/
│   └── BrainTumorSegmentation3dClient/
│       ├── ClientImpl.py       # Client implementation
│       ├── loading_utils.py    # Data loading utilities
│       └── utils.py            # Client-side utilities
│
├── server_visualize.py         # Server-side training visualisation
├── Resultsplots.py             # Results plotting and analysis
└── test.py                     # Testing and evaluation script
```

---

## Getting Started

### Prerequisites

```bash
pip install flwr torch monai
```

### Running the Server

```bash
python server_app.py
```

### Running a Client

```bash
python client_app.py --client-id <ID> --data-path <PATH>
```

---

## Method

FedWSOComp extends standard federated averaging with two complementary compression stages applied to client model updates before transmission:

1. **Sparsification** — only the top-k weights (by magnitude) are transmitted, zeroing out the rest.
2. **Clustering / Quantization** — remaining weights are quantized into clusters, reducing the bit-width of each update.

These stages are coordinated by the `FedWSOCompStrategy`, which handles aggregation and decompression on the server side.

---

## Dataset

This project uses the **FeTS 2022 / BraTS** dataset, which consists of multi-institutional multi-parametric MRI (mpMRI) scans of gliomas. Access requires registration at the official challenge page.

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{raza2026towards,
  title     = {Towards robust neurocomputing model in efficient federated brain tumour segmentation with sparsification and weights clustering},
  author    = {Raza, Asaf and others},
  journal   = {Neurocomputing},
  pages     = {133142},
  year      = {2026},
  publisher = {Elsevier}
}
```

> Raza, Asaf, et al. "Towards robust neurocomputing model in efficient federated brain tumour segmentation with sparsification and weights clustering." *Neurocomputing* (2026): 133142.


---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
