# %% set to the current directory
import os
import sys
import pickle
import gc
import itertools
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from layers import ClipModel, MLP, ProjectionHead, ResidualHead
from losses import ClipLoss
from plotting import plot_matrix
from training import (
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
    return np.stack(brain_features), text_embeddings_array[valid_indices]

# Load and Pair
preprocessed_train_brainiac, preprocessed_train_text_embeddings = align_and_load_brainiac(
    train_pmids, preprocessed_train_text_embeddings
)
preprocessed_test_brainiac, preprocessed_test_text_embeddings = align_and_load_brainiac(
    test_pmids, preprocessed_test_text_embeddings
)

# CRITICAL: Input Normalization for Transformer Features
# This ensures that the 768-dim ViT features are on the same scale as the text embeddings
scaler = StandardScaler()
preprocessed_train_brainiac = scaler.fit_transform(preprocessed_train_brainiac)
preprocessed_test_brainiac = scaler.transform(preprocessed_test_brainiac)
gc.collect()

# %% GRID SEARCH CONFIGURATION
device = "cuda" if torch.cuda.is_available() else "cpu"
brainiac_dim = 768
batch_size = 128
pilot_epochs = 40 
recall_fn = partial(recall_n, thresh=0.95, reduce_mean=True)

# Grid designed to force generalization via bottlenecking and high regularization
param_grid = {
    'latent_size': [256, 512],
    'dropout': [0.5, 0.7],
    'weight_decay': [0.1, 0.5, 1.0],
    'lr': [1e-4, 5e-5]
}

keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
print(f"🔍 Starting Grid Search: {len(combinations)} combinations...")

best_val_recall = -1
best_params = None
output_dir = Path(current_folder_path)

# Split indices for the search (using 1000 samples for validation)
train_idx, val_idx = next(KFold(n_splits=15).split(preprocessed_train_text_embeddings))
val_idx = val_idx[:1000]

# %% Execution Loop
for i, params in enumerate(combinations):
    print(f"\n--- 🧪 [{i+1}/{len(combinations)}] Testing: {params} ---")
    
    # Setup Loaders
    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(preprocessed_train_brainiac[train_idx]).float(),
        torch.from_numpy(preprocessed_train_text_embeddings[train_idx]).float()
    ), batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(TensorDataset(
        torch.from_numpy(preprocessed_train_brainiac[val_idx]).float(),
        torch.from_numpy(preprocessed_train_text_embeddings[val_idx]).float()
    ), batch_size=batch_size, shuffle=False)

    # Simplified architecture to combat the extreme overfitting observed
    model = ClipModel(
        image_model=nn.Sequential(
            ProjectionHead(brainiac_dim, params['latent_size'], dropout=params['dropout']),
            ResidualHead(params['latent_size'], dropout=params['dropout']),
        ),
        text_model=nn.Sequential(
            ProjectionHead(preprocessed_train_text_embeddings.shape[1], params['latent_size'], dropout=params['dropout']),
            ResidualHead(params['latent_size'], dropout=params['dropout']),
        ),
        logit_scale=10
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    
    # FIX: By passing callbacks=None, train() returns 3 values (model, train_loss, val_loss)
    # This prevents the "expected 4, got 3" or "expected 3, got 4" unpacking error.
    clip_model, train_loss, val_loss = train(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        ClipLoss(),    # criterion (5th arg)
        None,          # scheduler (6th arg)
        num_epochs=pilot_epochs, 
        device=device, 
        verbose=True, 
        output_dir=None, 
        callbacks=None  # Explicitly None to return 3 values
    )

    # Evaluate Validation Performance
    img_emb, txt_emb = predict(clip_model, val_loader, device=device)
    sim = (img_emb @ txt_emb.T).softmax(dim=1).numpy()
    current_val_recall = recall_fn(sim, np.eye(len(sim)), n_first=10)
    
    print(f"📊 Val Recall@10: {current_val_recall:.4f}")

    if current_val_recall > best_val_recall:
        best_val_recall = current_val_recall
        best_params = params
        torch.save(clip_model.state_dict(), output_dir / "grid_best_model.pt")

# %% FINAL SUMMARY
print("\n" + "="*40)
print(f"🏆 GRID SEARCH COMPLETE")
print(f"Best Recall@10: {best_val_recall:.4f}")
print(f"Best Parameters: {best_params}")
print("="*40)