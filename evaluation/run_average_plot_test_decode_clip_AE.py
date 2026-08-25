import os
import sys

# Path to the specific project directory
project_folder_path = '/data/parietal/store3/work/fghayyem/projects/MICCAI_2024/retreat-neuro-llm'
# Path to the parent directory
parent_folder_path = '/data/parietal/store3/work/fghayyem/projects/MICCAI_2024/'

# Append both directories to sys.path
sys.path.append(project_folder_path)
sys.path.append(parent_folder_path)

# Change the current working directory to the project directory
os.chdir(project_folder_path)
print("Current Working Directory: ", os.getcwd())
#%%

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import tqdm
from nilearn import image

from scripts.downstream_task.clip.utils import load_atlas_masked

#%% save paths

save_path = "/data/parietal/store3/work/fghayyem/projects/MICCAI_2024/retreat-neuro-llm/scripts/downstream_task/clip/clip/Results_DiceScore/test"


# file_name = "NeuroConText_AE_RandomModel.png"
# array_file_name = "NeuroConText_AE_RandomModel.npy"
# violin_file_name = "NeuroConText_AE_Violin_RandomModel.png"

# file_name = "NeuroConText_AE_beta0_alpha1.png"
# array_file_name = "NeuroConText_AE_beta0_alpha1.npy"
# violin_file_name = "NeuroConText_AE_Violin_beta0_alpha1.png"

# file_name = "NeuroConText_AE_beta1_alpha0.png"
# array_file_name = "NeuroConText_AE_beta1_alpha0.npy"
# violin_file_name = "NeuroConText_AE_Violin_beta1_alpha0.png"

file_name = "NeuroConText_AE.png"
array_file_name = "NeuroConText_AE.npy"
box_file_name = "NeuroConText_AE_box.png"


os.makedirs(save_path, exist_ok=True)
file_path = os.path.join(save_path, file_name)
array_file_path = os.path.join(save_path, array_file_name)
box_file_path = os.path.join(save_path, box_file_name)


#%%

def compute_dice(true_voxels, pred_voxels, thresholds):
    dices = []
    for threshold in thresholds:
        percentile_true = true_voxels > np.percentile(true_voxels, threshold)
        percentile_pred = pred_voxels > np.percentile(pred_voxels, threshold)

        dices.append(
            2*np.sum(np.logical_and(percentile_true, percentile_pred))
            / (np.sum(percentile_true) + np.sum(percentile_pred))
        )

    return dices


# %%
current_dir = Path(__file__).parent
# clip_dir = current_dir / "reconstruction_output"
clip_dir = current_dir.parent / "reconstruction_output"

# %%
groundtruth_test_difumo = pd.read_csv(clip_dir / "preprocessed_test_gaussian_embeddings_groundtruth_difumo.csv", index_col="pmid")
predicted_clip_org_test_difumo = pd.read_csv(clip_dir / "test_clip_AE_latent_decoded_difumo.csv", index_col="pmid")

#%%
masker, atlas, atlas_masked, mask = load_atlas_masked(dimension=groundtruth_test_difumo.shape[1])

# %%
def get_voxels(path, target_img):
    img = nib.load(path)
    img = image.resample_to_img(img, target_img)
    return img.get_fdata().flatten()

def get_voxels_from_difumo(array, target_img):
    img = masker.inverse_transform(array @ atlas_masked)
    img = image.resample_to_img(img, target_img)
    return img.get_fdata().flatten()

def get_voxels_from_difumo_test(array):
    img = masker.inverse_transform(array @ atlas_masked)
    # img = image.resample_to_img(img, target_img)
    return img.get_fdata().flatten()


dice_thresholds = [80, 85, 90, 95, 96, 97, 98, 99, 99.5, 99.8]
text2brain = []
neuroquery = []
clip = []

# for index, el in enumerate(tqdm.tqdm(groundtruth_test_difumo.to_dict("records"))):
for pmid in tqdm.tqdm(groundtruth_test_difumo.index):

    groundtruth_test_voxels = get_voxels_from_difumo_test(groundtruth_test_difumo.loc[pmid])
    predicted_test_voxels = get_voxels_from_difumo_test(predicted_clip_org_test_difumo.loc[pmid])

    
    
    clip_dices = compute_dice(
        groundtruth_test_voxels,
        predicted_test_voxels,
        dice_thresholds,
    )
    print(f"clip_dices: {clip_dices}")
    clip.append(clip_dices)
    print("CLIP: ", np.mean(clip, axis=0))
    del predicted_test_voxels

#%%
clip = np.array(clip)

clip_mean = np.mean(clip, axis=0)
clip_mean_rounded = np.round(clip_mean, 2)
print("CLIP: ", clip_mean_rounded)

np.save(array_file_path, clip)

# %%
import matplotlib.pyplot as plt

alpha = 0.2
font_size = 18  # Set the desired font size here

plt.plot(dice_thresholds, clip.mean(axis=0), label="NeuroConText - Original (separate decode)", c="red",marker='o')

plt.fill_between(
    dice_thresholds,
    clip.mean(axis=0) - clip.std(axis=0),
    clip.mean(axis=0) + clip.std(axis=0),
    alpha=alpha,
    color="red",
)


plt.xticks(dice_thresholds, [f"{t}" for t in dice_thresholds], fontsize=font_size, rotation=90)
plt.yticks(fontsize=font_size)
plt.xlabel("Threshold [%]", fontsize=font_size)  # Set x-axis label to "Threshold"
plt.ylabel("Dice score", fontsize=font_size)  # Set y-axis label to "Dice score"
plt.ylim(0, 1)  # Set y-axis limits
plt.title("Dice score as a function of threshold", fontsize=font_size)
# plt.legend(fontsize=font_size)
# plt.xscale('log')
plt.grid(True, which='both', axis='both', linestyle='--')  # Enable grid lines for both axes
# plt.xscale('log')

plt.savefig(file_path, bbox_inches='tight')
plt.show()

#%% Box plot

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

font_size = 24

cmap = plt.get_cmap('tab10')
colors = [cmap(i) for i in range(len(dice_thresholds))]

# Combine all data into a long format suitable for seaborn
data = []
for i, threshold in enumerate(dice_thresholds):
    print(f"i:{i}")
    print(f"threshold:{threshold}")
    for score in clip[:, i]:
        data.append((threshold, score))
data = np.array(data)

# Create a box plot
plt.figure(figsize=(6, 6))
boxplot = sns.boxplot(x=data[:, 0], y=data[:, 1], width=0.3)

# Calculate mean, median and std for each threshold
mean_std_median = []
unique_thresholds = np.unique(data[:, 0])

for i, threshold in enumerate(unique_thresholds):
    group_scores = data[data[:, 0] == threshold][:, 1].astype(float)
    mean = np.mean(group_scores)
    std = np.std(group_scores)
    median = np.median(group_scores)
    mean_std_median.append((threshold, mean, std, median))

# Create legend entries with color patches, ensuring unique thresholds
legend_entries = []
for i, (t, m, s, md) in enumerate(mean_std_median):
    # Get the color of the box corresponding to the threshold
    # color = boxplot.patches[i].get_facecolor()
    color = colors[i]
    # Create a legend entry with the statistics
    legend_entry = Patch(color=color, label=f' Mean={m:.2f}, Std={s:.2f}, Median={md:.2f}')
    legend_entries.append(legend_entry)

# Add the legend to the plot with the custom entries
plt.legend(handles=legend_entries, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=font_size)

plt.xlabel("Threshold", fontsize=font_size)
plt.ylabel("Dice score", fontsize=font_size)
# plt.ylim(0, 0.7)  # Set y-axis limits
plt.title("Box plot", fontsize=font_size)
plt.xticks(fontsize=font_size , rotation=90)
plt.yticks(fontsize=font_size)
plt.grid()

plt.savefig(box_file_path, bbox_inches='tight')
plt.show()


 # %% Load data

# import numpy as np
# Load the .npy file
# file_path = '/data/parietal/store3/work/fghayyem/projects/MICCAI_2024/retreat-neuro-llm/scripts/downstream_task/clip/clip/Results_DiceScore/test/NeuroConText_AE_RandomModel.npy'
# file_path = '/data/parietal/store3/work/fghayyem/projects/MICCAI_2024/retreat-neuro-llm/scripts/downstream_task/clip/clip/Results_DiceScore/test/NeuroConText_AE.npy'
# file_path = '/data/parietal/store3/work/fghayyem/projects/MICCAI_2024/retreat-neuro-llm/scripts/downstream_task/clip/clip/Results_DiceScore/test/NeuroConText_AE_beta0_alpha1.npy'
# file_path = '/data/parietal/store3/work/fghayyem/projects/MICCAI_2024/retreat-neuro-llm/scripts/downstream_task/clip/clip/Results_DiceScore/test/NeuroConText_AE_beta1_alpha0.npy'
# file_path = '/data/parietal/store3/work/fghayyem/projects/MICCAI_2024/retreat-neuro-llm/scripts/downstream_task/clip/clip/Results_DiceScore/test/NeuroConText_org.npy'

# clip = np.load(file_path)




# %%
