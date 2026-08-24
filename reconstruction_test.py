import os
import gc
import pickle
import torch
import torch.nn as nn
from pathlib import Path
from nilearn import datasets, plotting
from nilearn.maskers import NiftiMapsMasker

from layers import ClipModel_autoencoder, ProjectionHead, ResidualHead

# ---------------------------------------------------------
# 1. Setup Paths & Device
# ---------------------------------------------------------
current_directory = os.getcwd()
data_dir = os.path.join(current_directory, "data", "data_NeuroConText")
if not os.path.exists(data_dir):
    data_dir = os.path.join(current_directory, "data")

ckpt_path = Path(current_directory) / "output_full" / "best_val_full.pt"
output_dir = Path(current_directory) / "output_full" / "generated_brain_maps"
output_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
# 2. Load Only 'preprocessed_test_text_embeddings'
# ---------------------------------------------------------
pkl_path = os.path.join(data_dir, "preprocessed_test_text_embeddings.pkl")
print(f"📦 Loading {pkl_path}...")
with open(pkl_path, "rb") as f:
    preprocessed_test_text_embeddings = pickle.load(f)
gc.collect()
print("loaded.")

# ---------------------------------------------------------
# 3. Instantiate & Load Model Architecture
# ---------------------------------------------------------
output_size = 512                                             # DiFuMo dimension
text_dim = preprocessed_test_text_embeddings.shape[1]        # 4096
dropout = 0.6

model = ClipModel_autoencoder(
    image_model=nn.Sequential(
        ResidualHead(output_size, dropout=dropout),
        ResidualHead(output_size, dropout=dropout),
        ResidualHead(output_size, dropout=dropout),
    ),
    text_model=nn.Sequential(
        ProjectionHead(text_dim, output_size, dropout=dropout),
        ResidualHead(output_size, dropout=dropout),
        ResidualHead(output_size, dropout=dropout),
    ),
    decoder_model=nn.Sequential(
        ResidualHead(output_size, dropout=dropout),
        ResidualHead(output_size, dropout=dropout),
    ),
    logit_scale=10,
    logit_bias=None,
).to(device)

model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

# ---------------------------------------------------------
# 4. Sample 5 Text Embeddings & Predict DiFuMo Coefficients
# ---------------------------------------------------------
sample_indices = [0]
sample_text_tensors = torch.from_numpy(
    preprocessed_test_text_embeddings[sample_indices]
).float().to(device)

with torch.no_grad():
    # Encode text into shared latent space, then decode into 512 DiFuMo coefficients
    text_latents = model.text_model(sample_text_tensors)
    predicted_difumo = model.decoder_model(text_latents)
    predicted_coeffs = predicted_difumo.cpu().numpy()  # (5, 512)

# ---------------------------------------------------------
# 5. Inverse-Transform to 3D MNI Brain Maps via DiFuMo Atlas
# ---------------------------------------------------------
difumo = datasets.fetch_atlas_difumo(dimension=512, resolution_mm=2)
masker = NiftiMapsMasker(maps_img=difumo.maps).fit()

for i, (orig_idx, coeffs_1d) in enumerate(zip(sample_indices, predicted_coeffs)):
    brain_map_nii = masker.inverse_transform(coeffs_1d)

    nii_path = output_dir / f"test_sample_idx_{orig_idx}.nii.gz"
    plot_path = output_dir / f"test_sample_idx_{orig_idx}.png"

    brain_map_nii.to_filename(str(nii_path))

    plotting.plot_stat_map(
        brain_map_nii,
        title=f"Decoded DiFuMo Activation (Test Index {orig_idx})",
        display_mode="ortho",
        colorbar=True,
        threshold=0.01,
        output_file=str(plot_path)
    )

    print(f"Sample {orig_idx} saved:")
    print(f"  NIfTI -> {nii_path}")
    print(f"  Plot  -> {plot_path}")
