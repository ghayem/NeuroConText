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
import gc  # Garbage Collector

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

# %% Load test-set pmids (row-aligned with preprocessed_test_gaussian_embeddings)
# This file was generated separately to fix a pre-existing ordering issue --
# test_pmids.pkl (a set, unordered) cannot be used for this purpose.

with open(os.path.join(current_folder_path, 'pmids_order', 'test_pmids.txt')) as f:
    test_pmids = [int(line.strip()) for line in f]

assert len(test_pmids) == preprocessed_test_gaussian_embeddings.shape[0], (
    f"pmid count {len(test_pmids)} != test set size "
    f"{preprocessed_test_gaussian_embeddings.shape[0]}"
)

# %% import required modules

import shutil
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from layers import ClipModel_autoencoder, MLP, ProjectionHead, ResidualHead
from losses import ClipLoss
from plotting import plot_matrix
from training import (
    check_model_parameter_callback, count_parameters,
    diagonal_callback, non_diagonal_callback,
    predict_autoencoder, recall_n_callback, train_autoencoder,
)

from sklearn.preprocessing import Normalizer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from metrics import mix_match
from src.utils import plot_training, recall_n

# %% training the model (joint retrieval + reconstruction)
plot_verbose = True
batch_size = 128
lr_encoder = 1e-4
lr_decoder = 1e-4
weight_decay = 0.1
dropout = 0.6
num_epochs = 50

# Loss weighting: alpha scales the contrastive (retrieval) loss,
# beta scales the MSE (reconstruction) loss. Both are trained jointly.
alpha = 1e-0
beta = 1e-4

output_size = preprocessed_test_gaussian_embeddings.shape[1]

device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # ClipModel_autoencoder = image encoder + text encoder (retrieval, shared latent
    # space via InfoNCE/Clip loss) + decoder (reconstructs DiFuMo coefficients from
    # the text latent, trained with MSE)
    model = ClipModel_autoencoder(
        image_model=nn.Sequential(
            ResidualHead(output_size, dropout=dropout),
            ResidualHead(output_size, dropout=dropout),
            ResidualHead(output_size, dropout=dropout),
        ),
        text_model=nn.Sequential(
            ProjectionHead(preprocessed_train_text_embeddings.shape[1], output_size, dropout=dropout),
            ResidualHead(output_size, dropout=dropout),
            ResidualHead(output_size, dropout=dropout),
        ),
        decoder_model=nn.Sequential(
            ResidualHead(output_size, dropout=dropout),
            ResidualHead(output_size, dropout=dropout),
        ),
        **loss_specific_kwargs,
    )
    print(count_parameters(model))
    optimizer_encoder = torch.optim.AdamW(
        list(model.image_model.parameters()) + list(model.text_model.parameters()),
        lr=lr_encoder,
        weight_decay=weight_decay,
    )
    optimizer_decoder = torch.optim.AdamW(
        model.decoder_model.parameters(),
        lr=lr_decoder,
        weight_decay=weight_decay,
    )
    scheduler = None
    output_dir = Path(__file__).parent / "output_full"
    output_dir.mkdir(exist_ok=True)

    (
        clip_model,
        clip_train_loss,
        clip_val_loss,
        loss_contrastive_train,
        loss_contrastive_val,
        loss_mse_train,
        loss_mse_val,
        callback_outputs,
    ) = train_autoencoder(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_encoder=optimizer_encoder,
        optimizer_decoder=optimizer_decoder,
        scheduler=scheduler,
        criterion=criterion,
        alpha=alpha,
        beta=beta,
        num_epochs=num_epochs,
        device=device,
        verbose=True,
        output_dir=output_dir,
        callbacks=[
            # You can comment those callbacks to fasten the training
            # recall_n_callback(val_loader, n=10, device=device),
            # diagonal_callback(val_loader, device=device),
            # non_diagonal_callback(val_loader, device=device),
            # check_model_parameter_callback("logit_scale"),
            # check_model_parameter_callback("logit_bias"),
        ],
    )

    # Re-save the checkpoints with "full" in the name, so it's clear these
    # weights come from the joint retrieval+reconstruction model.
    last_ckpt = output_dir / "last.pt"
    best_val_ckpt = output_dir / "best_val.pt"
    last_full_ckpt = output_dir / "last_full.pt"
    best_val_full_ckpt = output_dir / "best_val_full.pt"
    if last_ckpt.exists():
        shutil.copy(last_ckpt, last_full_ckpt)
    if best_val_ckpt.exists():
        shutil.copy(best_val_ckpt, best_val_full_ckpt)

    if plot_verbose:
        callback_plot_kwargs = [
            {"ylabel": "Validation\nRecall@10", "color": "b", "ylim": [0, 1]},
            {"ylabel": "Diagonal Mean", "color": "b", "ylim": [1e-7, 1], "yscale": "log"},
            {"ylabel": "Non-diagonal Mean", "color": "b", "ylim": [1e-7, 1], "yscale": "log"},
            {"ylabel": "Logit scale", "color": "black"},
            {"ylabel": "Logit bias", "color": "black"},
        ]
        print("Total loss (contrastive + MSE)")
        plot_training(
            clip_train_loss,
            clip_val_loss,
            callback_outputs,
            callback_kwargs=callback_plot_kwargs,
        )
        print("Contrastive loss (retrieval)")
        plot_training(
            loss_contrastive_train,
            loss_contrastive_val,
            callback_outputs,
            callback_kwargs=callback_plot_kwargs,
        )
        print("MSE loss (reconstruction)")
        plot_training(
            loss_mse_train,
            loss_mse_val,
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
        ("train", small_train_loader, last_full_ckpt),
        ("validation", val_loader, best_val_full_ckpt),
        ("test", test_loader, best_val_full_ckpt),
    ]:
        clip_model.load_state_dict(torch.load(weights_path))
        clip_model.eval()

        # image_embeddings/text_embeddings live in the shared latent space (retrieval);
        # latent_decoded is the reconstructed DiFuMo coefficients predicted from text (reconstruction)
        image_embeddings, text_embeddings, latent_decoded = predict_autoencoder(clip_model, loader, device=device)

        similarity = (text_embeddings @ image_embeddings.T).softmax(dim=1).numpy()
        if plot_verbose:
            # Plot similarity matrices that should be diagonal
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
            gauss_similarity = (image_embeddings @ image_embeddings.T).numpy()
            plot_matrix(gauss_similarity[:100, :100], ax=axes[0], title="Gauss-to-Gauss")
            text_similarity = (text_embeddings @ text_embeddings.T).numpy()
            plot_matrix(text_similarity[:100, :100], ax=axes[1], title="Text-to-text")
            plot_matrix(similarity[:100, :100], ax=axes[2], title="Text-to-Gauss")
            fig.suptitle(f"Learnt similarities - {loader_name}")
            plt.tight_layout()
            plt.show()

        # --- Retrieval metrics ---
        nq_perf = recall_fn(similarity, np.eye(len(similarity)), n_first=10)
        nq_perf_100 = recall_fn(similarity, np.eye(len(similarity)), n_first=100)
        nq_perf_all = recall_fn(similarity, np.eye(len(similarity)), n_first=len(similarity))

        metrics[loader_name]["recall@10"].append(nq_perf)
        metrics[loader_name]["recall@100"].append(nq_perf_100)
        metrics[loader_name]["mix_match"].append(100 * mix_match(similarity))

        # --- Reconstruction metric ---
        # Reconstruction here is only about predicting the DiFuMo coefficients
        # (i.e. the image/gaussian embeddings), so we simply compare the decoded
        # latent against the ground-truth DiFuMo coefficients with MSE.
        ground_truth_difumo = torch.stack([loader.dataset[i][0] for i in range(len(loader.dataset))])
        reconstruction_mse = nn.functional.mse_loss(latent_decoded, ground_truth_difumo).item()
        metrics[loader_name]["reconstruction_mse"].append(reconstruction_mse)

        # --- Export test-set DiFuMo coefficients for downstream Dice evaluation ---
        # (evaluation/dice_eval_test.py reads these two CSVs from reconstruction_output/)
        if loader_name == "test":
            reconstruction_dir = Path(__file__).parent / "reconstruction_output"
            reconstruction_dir.mkdir(exist_ok=True)

            test_index = pd.Index(test_pmids, name="pmid")

            pd.DataFrame(latent_decoded.numpy(), index=test_index).to_csv(
                reconstruction_dir / "test_clip_AE_latent_decoded_difumo.csv"
            )
            pd.DataFrame(ground_truth_difumo.numpy(), index=test_index).to_csv(
                reconstruction_dir / "preprocessed_test_gaussian_embeddings_groundtruth_difumo.csv"
            )
            print(f"Test DiFuMo coefficients exported to {reconstruction_dir}")


print(f"Metrics after {fold} folds")
for loader_name in ["train", "validation", "test"]:
    print("="*10, loader_name, "="*10)
    for metric_name in ["recall@10", "recall@100", "mix_match", "reconstruction_mse"]:
        print(f"{metric_name}: {np.mean(metrics[loader_name][metric_name]):.3f} +- {np.std(metrics[loader_name][metric_name]):.3f}")

print(f"Checkpoints saved to: {output_dir}")
print(f"  last_full.pt -> {last_full_ckpt}")
print(f"  best_val_full.pt -> {best_val_full_ckpt}")
