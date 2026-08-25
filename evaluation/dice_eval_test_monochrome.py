#%% Setup paths

import os
import sys
from pathlib import Path

current_folder_path = os.getcwd()
parent_folder_path = os.path.dirname(current_folder_path)
sys.path.append(current_folder_path)
sys.path.append(parent_folder_path)

print("Current Working Directory: ", os.getcwd())

#%% Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from matplotlib.patches import Patch
from nilearn import datasets, image
from nilearn.maskers import NiftiMasker

#%% Save paths

save_path = Path("results") / "Results_DiceScore" / "test"
save_path.mkdir(parents=True, exist_ok=True)

file_name = "NeuroConText_AE.png"
array_file_name = "NeuroConText_AE.npy"
box_file_name = "NeuroConText_AE_box.png"

file_path = save_path / file_name
array_file_path = save_path / array_file_name
box_file_path = save_path / box_file_name

#%% Load ground truth and predicted DiFuMo coefficients

current_dir = Path(__file__).parent
clip_dir = current_dir.parent / "reconstruction_output"

groundtruth_test_difumo = pd.read_csv(
    clip_dir / "preprocessed_test_gaussian_embeddings_groundtruth_difumo.csv", index_col="pmid"
)
predicted_test_difumo = pd.read_csv(
    clip_dir / "test_clip_AE_latent_decoded_difumo.csv", index_col="pmid"
)

#%% Build atlas + mask from local nilearn cache (~/nilearn_data/difumo_atlases)
# Replaces the old load_atlas_masked(), which depended on a task_mask.nii.gz
# that isn't accessible from this machine/account. fetch_atlas_difumo reads
# the already-cached maps.nii.gz instead of downloading, since the expected
# directory structure (<dimension>/<resolution>mm/maps.nii.gz) is present.

dimension = groundtruth_test_difumo.shape[1]
resolution_mm = 2

atlas = datasets.fetch_atlas_difumo(dimension=dimension, resolution_mm=resolution_mm)

mni_mask = datasets.load_mni152_brain_mask()
mask = image.resample_to_img(mni_mask, atlas.maps, interpolation="nearest")

masker = NiftiMasker(mask_img=mask).fit()
atlas_masked = masker.transform(atlas.maps)  # shape: (dimension, n_voxels_in_mask)

#%% Helper functions

def compute_dice(true_voxels, pred_voxels, thresholds):
    dices = []
    for threshold in thresholds:
        percentile_true = true_voxels > np.percentile(true_voxels, threshold)
        percentile_pred = pred_voxels > np.percentile(pred_voxels, threshold)
        dices.append(
            2 * np.sum(np.logical_and(percentile_true, percentile_pred))
            / (np.sum(percentile_true) + np.sum(percentile_pred))
        )
    return dices


def get_voxels_from_difumo(array):
    img = masker.inverse_transform(array @ atlas_masked)
    return img.get_fdata().flatten()

#%% Compute dice scores per article, across thresholds

dice_thresholds = [95, 97, 98, 99]
clip_scores = []

for pmid in tqdm.tqdm(groundtruth_test_difumo.index):
    groundtruth_voxels = get_voxels_from_difumo(groundtruth_test_difumo.loc[pmid])
    predicted_voxels = get_voxels_from_difumo(predicted_test_difumo.loc[pmid])

    clip_scores.append(compute_dice(groundtruth_voxels, predicted_voxels, dice_thresholds))
    del predicted_voxels

clip_scores = np.array(clip_scores)
np.save(array_file_path, clip_scores)

print("Mean dice per threshold:", np.round(clip_scores.mean(axis=0), 2))

#%% Line plot: mean dice +/- std vs threshold

font_size = 18

plt.figure()
plt.plot(
    dice_thresholds,
    clip_scores.mean(axis=0),
    label="NeuroConText - Original (separate decode)",
    c="red",
    marker="o",
)
plt.fill_between(
    dice_thresholds,
    clip_scores.mean(axis=0) - clip_scores.std(axis=0),
    clip_scores.mean(axis=0) + clip_scores.std(axis=0),
    alpha=0.2,
    color="red",
)
plt.xticks(dice_thresholds, [f"{t}" for t in dice_thresholds], fontsize=font_size, rotation=90)
plt.yticks(fontsize=font_size)
plt.xlabel("Threshold [%]", fontsize=font_size)
plt.ylabel("Dice score", fontsize=font_size)
plt.ylim(0, 1)
plt.title("Dice score as a function of threshold", fontsize=font_size)
plt.grid(True, which="both", axis="both", linestyle="--")
plt.savefig(file_path, bbox_inches="tight")
plt.show()

#%% Box plot per threshold

font_size = 24

# --- Color setting ---
# Set to False to color every box distinctly (tab10 palette).
# Set to True to make all boxes the same color instead.
use_single_color = True
single_color = "red"

cmap = plt.get_cmap("tab10")
if use_single_color:
    colors = [single_color] * len(dice_thresholds)
else:
    colors = [cmap(i) for i in range(len(dice_thresholds))]

data = np.array(
    [
        (threshold, score)
        for i, threshold in enumerate(dice_thresholds)
        for score in clip_scores[:, i]
    ]
)

plt.figure(figsize=(6, 6))
ax = sns.boxplot(x=data[:, 0], y=data[:, 1], width=0.3, hue=data[:, 0], palette=colors)

# hue was only used to get per-box colors from `palette`; drop the
# auto-generated hue legend since we build our own stats legend below.
if ax.get_legend() is not None:
    ax.get_legend().remove()

# Belt-and-suspenders: explicitly set each box's facecolor so it's guaranteed
# to match the legend patches below, regardless of seaborn version quirks.
for patch, color in zip(ax.patches, colors):
    patch.set_facecolor(color)

legend_entries = []
for i, threshold in enumerate(dice_thresholds):
    group_scores = clip_scores[:, i]
    mean, std, median = group_scores.mean(), group_scores.std(), np.median(group_scores)
    legend_entries.append(
        Patch(color=colors[i], label=f"Mean={mean:.2f}, Std={std:.2f}, Median={median:.2f}")
    )

plt.legend(handles=legend_entries, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=font_size)
plt.xlabel("Threshold", fontsize=font_size)
plt.ylabel("Dice score", fontsize=font_size)
plt.title("Box plot", fontsize=font_size)
plt.xticks(fontsize=font_size, rotation=90)
plt.yticks(fontsize=font_size)
plt.grid()
plt.savefig(box_file_path, bbox_inches="tight")
plt.show()
