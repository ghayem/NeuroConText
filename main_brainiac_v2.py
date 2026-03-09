# %% set to the current directory
import os
import sys
import pickle
import gc
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from layers import ClipModel, MLP, ProjectionHead, ResidualHead
from losses import ClipLoss
from plotting import plot_matrix
from training import (
    check_model_parameter_callback,
    count_parameters,
    predict,
    train,
)
from metrics import mix_match
from src.utils import plot_training, recall_n

# Setup paths
current_folder_path = os.getcwd()
parent_folder_path = os.path.dirname(current_folder_path)
sys.path.extend([current_folder_path, parent_folder_path])
os.chdir(current_folder_path)

# %% Load metadata and text embeddings (Memory Optimized)
data_dir = os.path.join(current_folder_path, "data", "data_NeuroConText")
if not os.path.exists(data_dir):
    data_dir = os.path.join(current_folder_path, "data")

print(f"🚀 Loading base data from {data_dir}...")
for file in os.listdir(data_dir):
    if file.endswith(".pkl"):
        var_name = file.replace(".pkl", "")
        with open(os.path.join(data_dir, file), "rb") as f:
            globals()[var_name] = pickle.load(f)
        gc.collect()

# %% Load BrainIAC Embeddings and ALIGN with Text
brainiac_dir = "embeddings/npy"
print(f"🧠 Aligning BrainIAC features from {brainiac_dir}...")

def align_and_load_brainiac(pmid_list, text_embeddings_array):
    valid_indices = []
    brain_features = []

    for i, pmid in enumerate(pmid_list):
        path = os.path.join(brainiac_dir, f"pmid_{pmid}_embedding.npy")
        if os.path.exists(path):
            brain_features.append(np.load(path))
            valid_indices.append(i)

    if not brain_features:
        raise FileNotFoundError(f"No matching .npy files found in {brainiac_dir}")

    aligned_brain = np.stack(brain_features)
    aligned_text = text_embeddings_array[valid_indices]

    print(f"✅ Successfully paired {len(valid_indices)}/{len(pmid_list)} samples.")
    return aligned_brain, aligned_text

preprocessed_train_brainiac, preprocessed_train_text_embeddings = (
    align_and_load_brainiac(train_pmids, preprocessed_train_text_embeddings)
)
preprocessed_test_brainiac, preprocessed_test_text_embeddings = align_and_load_brainiac(
    test_pmids, preprocessed_test_text_embeddings
)
gc.collect()

# %% training configuration
plot_verbose = True
batch_size = 128
lr = 1e-4
weight_decay = 0.1
dropout = 0.4       # Reduced dropout (from 0.6) to help model capture complex features 
num_epochs = 100    # Increased epochs (from 50) to ensure convergence 
latent_size = 768   # Increased latent space to match BrainIAC (768) 
brainiac_dim = 768  # BrainIAC ViT features dimension

device = "cuda" if torch.cuda.is_available() else "cpu"
criterion = ClipLoss()
loss_specific_kwargs = {"logit_scale": 10}

test_dataset = TensorDataset(
    torch.from_numpy(preprocessed_test_brainiac).float(),
    torch.from_numpy(preprocessed_test_text_embeddings).float(),
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %% Training Loop
recall_fn = partial(recall_n, thresh=0.95, reduce_mean=True)
validation_size = 1000
k_fold = KFold(n_splits=len(preprocessed_train_text_embeddings) // validation_size)

metrics = {"train": defaultdict(list), "validation": defaultdict(list), "test": defaultdict(list)}
number_of_folds_to_run = 1

for fold, (train_index, val_index) in enumerate(k_fold.split(preprocessed_train_text_embeddings)):
    val_index = val_index[:validation_size]
    if fold >= number_of_folds_to_run:
        break

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(preprocessed_train_brainiac[train_index]).float(),
            torch.from_numpy(preprocessed_train_text_embeddings[train_index]).float(),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(preprocessed_train_brainiac[val_index]).float(),
            torch.from_numpy(preprocessed_train_text_embeddings[val_index]).float(),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    # UPDATED MODEL: Use 768 as the shared dimension
    model = ClipModel(
        image_model=nn.Sequential(
            # Since BrainIAC is already 768, we map 768 -> 768 [cite: 166, 174]
            ProjectionHead(brainiac_dim, latent_size, dropout=dropout),
            ResidualHead(latent_size, dropout=dropout),
            ResidualHead(latent_size, dropout=dropout),
        ),
        text_model=nn.Sequential(
            # Map Mistral (4096) -> 768 [cite: 167]
            ProjectionHead(preprocessed_train_text_embeddings.shape[1], latent_size, dropout=dropout),
            ResidualHead(latent_size, dropout=dropout),
            ResidualHead(latent_size, dropout=dropout),
        ),
        **loss_specific_kwargs,
    ).to(device)

    print(f"Total Parameters: {count_parameters(model)}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    output_dir = Path(__file__).parent

    clip_model, clip_train_loss, clip_val_loss, callback_outputs = train(
        model, train_loader, val_loader, optimizer, criterion, None,
        num_epochs, device, verbose=True, output_dir=output_dir, callbacks=[],
    )

    if plot_verbose:
        plot_training(clip_train_loss, clip_val_loss, callback_outputs,
                      callback_kwargs=[{"ylabel": "Recall@10", "color": "b", "ylim": [0, 1]}])

    for loader_name, loader, weights_path in [
        ("train", train_loader, output_dir / "last.pt"),
        ("validation", val_loader, output_dir / "best_val.pt"),
        ("test", test_loader, output_dir / "best_val.pt"),
    ]:
        clip_model.load_state_dict(torch.load(weights_path))
        image_embeddings, text_embeddings = predict(clip_model, loader, device=device)
        similarity = (image_embeddings @ text_embeddings.T).softmax(dim=1).numpy()

        metrics[loader_name]["recall@10"].append(recall_fn(similarity, np.eye(len(similarity)), n_first=10))
        metrics[loader_name]["recall@100"].append(recall_fn(similarity, np.eye(len(similarity)), n_first=100)) 
        metrics[loader_name]["mix_match"].append(100 * mix_match(similarity)) 

# %% Final Summary
print(f"\nFinal Metrics (BrainIAC Features):")
for loader_name in ["train", "validation", "test"]:
    print(f"--- {loader_name} ---")
    for m in ["recall@10", "recall@100", "mix_match"]:
        print(f"{m}: {np.mean(metrics[loader_name][m]):.3f}")