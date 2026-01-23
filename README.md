Here is the updated README. I have integrated the **uv** environment ## NeuroConText: Contrastive Text-to-Brain Mapping for Neuroscientific Literature

This repository contains the code for the paper accepted at MICCAI'24:

[NeuroConText paper at MICCAI'24](https://link.springer.com/chapter/10.1007/978-3-031-72384-1_31).

[NeuroConText paper extended version](https://www.biorxiv.org/content/10.1101/2025.05.23.655707v1.abstract).

[NeuroConText Supplementary Material](https://drive.google.com/file/d/17IJ7Jn9cHXbMiBEzCnTepDcleeXHpRN-/view?usp=drive_link).

---

### Getting Started

Follow these steps to set up the environment, download the data, and run the training pipeline.

### 1. Environment Setup (using `uv`)

We use [uv](https://github.com/astral-sh/uv) for extremely fast and reproducible dependency management.

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral-sh.uv.install.sh | sh

```


2. **Initialize the environment**:
From the project root, run the following to create a virtual environment with the correct Python version (3.10.9) and install all dependencies:
```bash
uv sync

```


3. **Activate the environment**:
```bash
source .venv/bin/activate

```



### 2. Download and Prepare Data

To automate the setup of the ~8GB dataset from Zenodo, we provide a high-performance parallel downloader. This script handles the download, extraction, and directory placement automatically.

```bash
# This script uses pycurl for parallel downloading and unzips to ../data/
uv run utils/download_data.py

```

### 3. Running the Code

Once the environment is synced and the data is downloaded, you can start the training and evaluation pipeline:

```bash
uv run main.py

```

---

### Directory Structure

```text
NeuroConText/
│
├── data/               # Automatically populated by download_data.py
│   └── data_NeuroConText/
│       └── (Extracted .pkl embedding files)
│
├── src/                # Core utility logic
│   ├── utils.py
│   └── ...
├── utils/
│   └── download_data.py # High-performance parallel downloader
│
├── layers.py           # Model architectures (ClipModel, ResidualHead)
├── losses.py           # Contrastive Loss functions
├── main.py             # Main entry point for training/evaluation
├── metrics.py          # Evaluation metrics (Recall@N, Mix-Match)
├── plotting.py         # Visualization utilities
├── training.py         # Training loops and callbacks
└── README.md

```

### Contact

For any issues or questions regarding the code, please contact fateme[dot]ghayem[at]gmail[dot]come.

### License

This work is supported by the KARAIB AI chair (ANR-20-CHIA-0025-01), the ANR-22-PESN-0012 France 2030 program, and the HORIZON-INFRA-2022-SERV-B-01 EBRAINS 2.0 infrastructure project.

---

Thank you for using NeuroConText!