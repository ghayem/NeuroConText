#%% set to the current directory

import os
import sys

# Get the current working directory
current_folder_path = os.getcwd()

# Get the parent directory
parent_folder_path = os.path.dirname(current_folder_path)

# Append both directories to sys.path
sys.path.append(current_folder_path)
sys.path.append(parent_folder_path)

# Change the current working directory to the current directory (optional, since it's already the current directory)
os.chdir(current_folder_path)

print("Current Working Directory: ", os.getcwd())
print("sys.path: ", sys.path)

# %% Load data (Memory Optimized)
import os
import pickle
import gc # Garbage Collector

current_directory = os.getcwd()
data_dir = os.path.join(current_directory, 'data', 'data_NeuroConText')

if not os.path.exists(data_dir):
    print(f"❌ Error: Directory not found at {data_dir}")
    data_dir = os.path.join(current_directory, 'data')

print(f"🚀 Loading data from {data_dir}...")

# Load files one by one directly into globals to save RAM
for file in os.listdir(data_dir):
    if file.endswith('.pkl'):
        var_name = file.replace('.pkl', '')
        file_path = os.path.join(data_dir, file)
        
        print(f"📦 Loading {var_name}...")
        with open(file_path, 'rb') as f:
            # Direct injection into globals avoids the 'loaded_data' dictionary overhead
            globals()[var_name] = pickle.load(f)
        
        # Force garbage collection after each large file load
        gc.collect()

print("✅ Data loaded. Memory freed.")

# %% import required modules

from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from layers import ClipModel, MLP, ProjectionHead, ResidualHead
from losses import ClipLoss
from plotting import plot_matrix
from training import (
    check_model_parameter_callback, count_parameters,
    diagonal_callback, non_diagonal_callback,
    predict, recall_n_callback, train,
)

from sklearn.preprocessing import Normalizer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from metrics import mix_match
from src.utils import plot_training, recall_n

# %% training the model
plot_verbose = True
batch_size = 128
lr = 1e-4
weight_decay = 0.1
dropout = 0.6
num_epochs = 50
output_size = preprocessed_test_gaussian_embeddings.shape[1]

device = "cuda" if torch.cuda.is_available() else "cpu"

# criterion = SigLipLoss()
criterion = ClipLoss()
is_clip_loss = criterion.__class__ == ClipLoss
loss_specific_kwargs = {
    "logit_scale": 10 if is_clip_loss else np.log(10),
    "logit_bias": None if is_clip_loss else -10,
}

test_dataset = TensorDataset(
    torch.from_numpy(preprocessed_test_gaussian_embeddings).float(),
    torch.from_numpy(preprocessed_test_text_embeddings).float(),
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
recall_fn = partial(recall_n, thresh=0.95, reduce_mean=True)

print(f"Using device: {device}")
validation_size = 1000
k_fold = KFold(n_splits=len(preprocessed_train_text_embeddings) // validation_size)

metrics = {
    "train": defaultdict(list),
    "validation": defaultdict(list),
    "test": defaultdict(list),
}
number_of_folds_to_run = 1
for fold, (train_index, val_index) in enumerate(k_fold.split(preprocessed_train_text_embeddings)):
    val_index = val_index[:validation_size]  # Strict 1000 validation samples
    if fold >= number_of_folds_to_run:
        break

    train_dataset = TensorDataset(
        torch.from_numpy(preprocessed_train_gaussian_embeddings[train_index]).float(),
        torch.from_numpy(preprocessed_train_text_embeddings[train_index]).float(),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(
        torch.from_numpy(preprocessed_train_gaussian_embeddings[val_index]).float(),
        torch.from_numpy(preprocessed_train_text_embeddings[val_index]).float(),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ClipModel(
        image_model=nn.Sequential(
            # ProjectionHead(preprocessed_train_gaussian_embeddings.shape[1], output_size, dropout=dropout),
            ResidualHead(output_size, dropout=dropout),
            ResidualHead(output_size, dropout=dropout),
            ResidualHead(output_size, dropout=dropout),
            # ResidualHead(output_size, dropout=dropout),
        ),
        text_model=nn.Sequential(
            ProjectionHead(preprocessed_train_text_embeddings.shape[1], output_size, dropout=dropout),
            ResidualHead(output_size, dropout=dropout),
            ResidualHead(output_size, dropout=dropout),
            # ResidualHead(output_size, dropout=dropout),
        ),
        **loss_specific_kwargs,
    )
    print(count_parameters(model))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = None  # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*num_epochs)
    output_dir = Path(__file__).parent

    clip_model, clip_train_loss, clip_val_loss, callback_outputs = train(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=num_epochs,
        device=device,
        verbose=True,
        output_dir=output_dir,
        # clip_grad_norm=0.3,
        callbacks=[
            # You can comment those callbacks to fasten the training
            # they are here to help understand what is happening across epochs
            # recall_n_callback(val_loader, n=10, device=device),
            # diagonal_callback(val_loader, device=device),
            # non_diagonal_callback(val_loader, device=device),
            # check_model_parameter_callback("logit_scale"),
            # check_model_parameter_callback("logit_bias"),
        ],
    )

    if plot_verbose:
        callback_plot_kwargs = [
            {"ylabel": "Validation\nRecall@10", "color": "b", "ylim": [0, 1]},
            {"ylabel": "Diagonal Mean", "color": "b", "ylim": [1e-7, 1], "yscale": "log"},
            {"ylabel": "Non-diagonal Mean", "color": "b", "ylim": [1e-7, 1], "yscale": "log"},
            {"ylabel": "Logit scale", "color": "black"},
            {"ylabel": "Logit bias", "color": "black"},
        ]
        plot_training(
            clip_train_loss,
            clip_val_loss,
            callback_outputs,
            callback_kwargs=callback_plot_kwargs,
        )

    # Define a small train dataset to get metrics faster
    small_train_dataset = TensorDataset(
        torch.from_numpy(preprocessed_train_gaussian_embeddings[train_index][:1000]).float(),
        torch.from_numpy(preprocessed_train_text_embeddings[train_index][:1000]).float(),
    )
    small_train_loader = DataLoader(small_train_dataset, batch_size=batch_size, shuffle=False)
    for loader_name, loader, weights_path in [
        ("train", small_train_loader, output_dir / "last.pt"),
        ("validation", val_loader, output_dir / "best_val.pt"),
        ("test", test_loader, output_dir / "best_val.pt"),
    ]:
        clip_model.load_state_dict(torch.load(weights_path))

        image_embeddings, text_embeddings = predict(clip_model, loader, device=device)
        similarity = (image_embeddings @ text_embeddings.T).softmax(dim=1).numpy()
        if plot_verbose:
            # Plot similarity matrices that should be diagonal
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
            gauss_similarity = (image_embeddings @ image_embeddings.T).numpy()
            plot_matrix(gauss_similarity[:100, :100], ax=axes[0], title="Gauss-to-Gauss")
            text_similarity = (text_embeddings @ text_embeddings.T).numpy()
            plot_matrix(text_similarity[:100, :100], ax=axes[1], title="Text-to-text")
            plot_matrix(similarity[:100, :100], ax=axes[2], title="Gauss-to-Text")
            fig.suptitle(f"Learnt similarities - {loader_name}")
            plt.tight_layout()
            plt.show()

        random_perf = 10 / len(similarity)

        nq_perf = recall_fn(similarity, np.eye(len(similarity)), n_first=10)
        nq_perf_100 = recall_fn(similarity, np.eye(len(similarity)), n_first=100)
        nq_perf_all = recall_fn(similarity, np.eye(len(similarity)), n_first=len(similarity))

        metrics[loader_name]["recall@10"].append(nq_perf)
        metrics[loader_name]["recall@100"].append(nq_perf_100)
        metrics[loader_name]["mix_match"].append(100*mix_match(similarity))


print(f"Metrics after {fold} folds")
for loader_name in ["train", "validation", "test"]:
    print("="*10, loader_name, "="*10)
    for metric_name in ["recall@10", "recall@100", "mix_match"]:
        print(f"{metric_name}: {np.mean(metrics[loader_name][metric_name]):.3f} +- {np.std(metrics[loader_name][metric_name]):.3f}")
