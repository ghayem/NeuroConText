## NeuroConText: Contrastive Text-to-Brain Mapping for Neuroscientific Literature

This repository contains the code for the paper accepted at MICCAI'24:

[NeuroConText paper at MICCAI'24](https://link.springer.com/chapter/10.1007/978-3-031-72384-1_31).

[NeuroConText paper extended version at Imaging Neuroscience, MIT Press, 2026](https://direct.mit.edu/imag/article/doi/10.1162/IMAG.a.1162/135353/NeuroConText-Contrastive-Learning-for-Neuroscience?searchresult=1).

[NeuroConText Supplementary Material](https://mitp.silverchair-cdn.com/mitp/content_public/journal/imag/jam/10.1162_imag.a.1162/2/imag.a.1162_supp.pdf?Expires=1774531548&Signature=Z20WiuTvdl6-wdlMQ3Np4e8FMsOzmXZTNVkzrcelLmvI8zdEyhfOojyBKzbOtJ3PNsqPa7zBV4w~4S~EZ6jAGtzSnZaYsPBrPSRHEBhs1p7FzRZwvr4HVxER4GNunqKhO9vy4wx1pTbZZxYztJLur69H00igN3zno0jYD1ekOVWf9du31h771Vu1BRA8ZqekeRfdmS1~eL-iV3xCVh6pEuT9~pxML5DiUC3odxTej4N9xveB0pEqEyj6o63VItm~pmuk6XMHlJVHvjp5JGO0kr4bYZsjTT0vUFbBxfld7Lkc-uOpcTswJnYI~5Djsy40ZfIHeBGmXCEuHB4QQgmyXQ__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA).

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

## Citation

Thank you for using NeuroConText! 
If you use this code, please cite:

```bibtex
@article{ghayem2026neurocontext,
  title={NeuroConText: Contrastive learning for neuroscience meta-analysis with rich text representation},
  author={Ghayem, Fateme and Meudec, Rapha{\"e}l and Dock{\`e}s, J{\'e}r{\^o}me and Thirion, Bertrand and Wassermann, Demian},
  journal={Imaging Neuroscience},
  volume={4},
  pages={IMAG--a},
  year={2026},
  publisher={MIT Press 255 Main Street, 9th Floor, Cambridge, Massachusetts 02142, USA~…}
}

@inproceedings{meudec2024neurocontext,
  title={NeuroConText: Contrastive text-to-brain mapping for neuroscientific literature},
  author={Meudec, Rapha{\"e}l and Ghayem, Fateme and Dock{\`e}s, J{\'e}r{\^o}me and Wassermann, Demian and Thirion, Bertrand},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={325--335},
  year={2024},
  organization={Springer}
}
```
