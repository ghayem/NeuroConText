## NeuroConText: Contrastive Text-to-Brain Mapping for Neuroscientific Literature

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
Create the virtual environment and install all dependencies:
```bash
uv sync

```


3. **Activate the environment**:
```bash
source .venv/bin/activate

```



### 2. Download and Prepare Data

We provide a high-performance parallel downloader to handle the ~8GB dataset from Zenodo. This script automates the download, extraction, and directory placement.

```bash
# Uses pycurl for parallel downloading; extracts to the data/ folder
uv run utils/download_data.py

```

### 3. Running the Code

Once the environment is synced and the data is downloaded, execute the training pipeline:

```bash
uv run main.py

```

---

### Directory Structure
```

NeuroConText/
│
├── data/                # Populated by download_data.py
│   └── data_NeuroConText/
│       └── (Extracted .pkl files)
│
├── src/                 # Core utilities
│   └── utils.py
│
├── utils/
│   └── download_data.py  # Parallel downloader
│
├── layers.py            # Model architectures
├── losses.py            # Contrastive losses
├── main.py              # Training entry point
├── metrics.py           # Evaluation logic
├── plotting.py          # Visualizations
├── training.py          # Training loop
└── README.md
```
---

### Contact

For any issues or questions regarding the code, please contact fateme[dot]ghayem[at]gmail[dot]com.

---

### License

This work is supported by the KARAIB AI chair (ANR-20-CHIA-0025-01), the ANR-22-PESN-0012 France 2030 program, and the HORIZON-INFRA-2022-SERV-B-01 EBRAINS 2.0 infrastructure project.

---

Thank you for using NeuroConText!
