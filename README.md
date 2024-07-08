# README

## NeuroConText: Contrastive Text-to-Brain Mapping for Neuroscientific Literature

### Introduction

This repository contains the code for the paper accepted at MICCAI'24:
**NeuroConText: Contrastive Text-to-Brain Mapping for Neuroscientific Literature**.

### Getting Started

To get started with this project, follow the steps below.

### Prerequisites

Make sure you have Python installed on your system. The code has been tested with Python 3.10.9. 

### Data

To work with the code, you need to download the dataset from the following link: 
[https://zenodo.org/uploads/12684666](https://zenodo.org/uploads/12684666).

After downloading, save the data in the `data` folder within the working directory of this project.

### Running the Code

Once the data is in place, you can run the main script to execute the code. Use the following command:

```bash
python main.py
```

### Directory Structure

The directory structure of the project should look like this:

```
NeuroConText/
│
├── data/
│   └── (Downloaded data files from [https://zenodo.org/uploads/12684666](https://zenodo.org/uploads/12684666))
│
├── src/
│   ├── __init__.py
│   ├── cognitive_atlas.py
│   ├── constants.py
│   ├── datasets.py
│   ├── embeddings.py
│   ├── load_trained_model.py
│   ├── loader.py
│   ├── metrics.py
│   ├── nnod.py
│   ├── parallel.py
│   └── utils.py
│
├── layers.py
├── losses.py
├── main.py
├── metrics.py
├── plotting.py
├── training.py
└── utils.py
├── README.md

```

### Contact

For any issues or questions regarding the code, please contact fateme[dot]ghayem[at]gmail[dot]come.

### License

This work is supported by the KARAIB AI chair (ANR-20-CHIA-0025-01), the ANR-22-PESN-0012 France 2030 program, and the HORIZON-INFRA-2022-SERV-B-01 EBRAINS 2.0 infrastructure project.

---

Thank you for using NeuroConText!
