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
import pickle
import nilearn
from nilearn import image

from scripts.downstream_task.clip.utils import load_atlas_masked

import matplotlib.pyplot as plt

#%%

df_name = "body" # you should set either you want to work with title, abstract, or body

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

file_name = "NeuroConText_org.png"
array_file_name = "NeuroConText_org.npy"
box_file_name = "NeuroConText_org_box.png"


os.makedirs(save_path, exist_ok=True)
file_path = os.path.join(save_path, file_name)
array_file_path = os.path.join(save_path, array_file_name)
box_file_path = os.path.join(save_path, box_file_name)

#%%

#%%

# Load the NQ KDEs to get the NQ pmids
file_path_kde_nq = '/data/parietal/store3/work/fghayyem/projects/MICCAI_2024/results/coordinates_KDE/neuroquery/brain_maps_nq.pkl'
file_path_kde_nq_pubget_merged = '/data/parietal/store3/work/fghayyem/projects/MICCAI_2024/results/coordinates_KDE/kde_merged_nq_pubget/merged_kde_brain_maps.pkl'

file_path_kde_groundtruth = '/data/parietal/store3/work/fghayyem/projects/MICCAI_2024/retreat-neuro-llm/scripts/downstream_task/clip/kde_brain_maps.pkl'

# with open(file_path_kde_nq, 'rb') as file:
#     # Load the content of the file into a variable
#     kde_nq = pickle.load(file)

# with open(file_path_kde_nq_pubget_merged, 'rb') as file:
#     # Load the content of the file into a variable
#     kde_nq_pubget_merged = pickle.load(file)

### This is the right groundtruth
with open(file_path_kde_groundtruth, 'rb') as file:
    # Load the content of the file into a variable
    kde_groundtruth = pickle.load(file)

# pmids_neuroquery = kde_nq.index
# pmids_neuroquery_pubget_merged = kde_nq_pubget_merged.index
pmids_kde_groundtruth = kde_groundtruth['kde_brain_maps_pmids']


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
predicted_clip_org_test_difumo = pd.read_csv(clip_dir / "test_clip_org_latent_decoded_difumo.csv", index_col="pmid")

#%%
masker, atlas, atlas_masked, mask = load_atlas_masked(dimension=groundtruth_test_difumo.shape[1])

#%%
from nilearn.input_data import NiftiMasker

path_masker_NQ = '/data/parietal/store3/work/fghayyem/projects/MICCAI_2024/retreat-neuro-llm/scripts/downstream_task/clip/clip/baselines/mask_img.nii'
masker_NQ = NiftiMasker(path_masker_NQ).fit()


# %%
def get_voxels(path, target_img):
    img = nib.load(path)
    img = image.resample_to_img(img, target_img)
    img = nilearn.image.smooth_img(img, fwhm=9) # smooth image
    return img.get_fdata().flatten()

def get_voxels_from_difumo(array, target_img):
    img = masker.inverse_transform(array @ atlas_masked)
    img = nilearn.image.smooth_img(img, fwhm=9) # smooth image
    img = image.resample_to_img(img, target_img)
    # return img.get_fdata().flatten()
    return img.get_fdata()

def get_voxels_from_difumo_test(array):
    img = masker.inverse_transform(array @ atlas_masked)
    img = nilearn.image.smooth_img(img, fwhm=9) # smooth image
    # img = image.resample_to_img(img, target_img)
    # return img.get_fdata().flatten()
    return img

def get_voxels_from_difumo__3D_nifti(array, target_img):
    img = masker.inverse_transform(array @ atlas_masked)
    img = nilearn.image.smooth_img(img, fwhm=9) # smooth image
    img = image.resample_to_img(img, target_img)
    # return img.get_fdata()
    return img

#%%
def get_brain_map_by_pmid(kde_groundtruth, pmid):
    # Extract PMIDs and brain maps from the dictionary
    brain_maps = kde_groundtruth['kde_brain_maps']  # Tuple of brain maps
    brain_pmids = kde_groundtruth['kde_brain_maps_pmids']
    brain_pmids_list = [pmid for sublist in brain_pmids for pmid in sublist]
    
    if pmid in brain_pmids_list:
        index = brain_pmids_list.index(pmid)
        # print(f"PMID {pmid} found at index {index} in the flattened list.")

        return brain_maps[index][0]
    
    else:
        print(f"{pmid} not found in pmid_list.")



# %% Plot actual kde groundtruth with pmid
import nibabel as nib
from nilearn import plotting

# Function to normalize the image
def normalize_img(img):
    if isinstance(img, str) or isinstance(img, Path):
        img = nib.load(img)
    data = img.get_fdata()
    data = (data - data.mean()) / data.std()
    return nib.Nifti1Image(data, img.affine)

# List of PMIDs from the image
sample_pmids = [32591929, 
                31114477, 
                31691912, 
                31711031, 
                35874153, 
                30488645]

# Plot parameters
vmax = 6
plot_parameters = {
    "cmap": "bwr",
    "vmax": vmax,
    "views": ["lateral"],
    "colorbar": True,
    "darkness": .3,
    "alpha": 1,
    "bg_on_data": True,
}

# Loop through each PMID and plot with title
for pmid in sample_pmids:
    groundtruth_sample_voxels_3D = get_brain_map_by_pmid(kde_groundtruth, pmid)
    # Plot with title including PMID
    plotting.plot_img_on_surf(
        normalize_img(groundtruth_sample_voxels_3D),
        title=f"PMID: {pmid} - Groundtruth Map",
        **plot_parameters,
    )
    plt.suptitle(f"PMID: {pmid} - Groundtruth Map", fontsize=24)
    plotting.show()



# %% Groundtruth from difumo reconstruction


import nibabel as nib
from nilearn import plotting

groundtruth_test_difumo = pd.read_csv(clip_dir / "preprocessed_test_gaussian_embeddings_groundtruth_difumo.csv", index_col="pmid")
groundtruth_train_difumo = pd.read_csv(clip_dir / "preprocessed_train_gaussian_embeddings_groundtruth_difumo.csv", index_col="pmid")

# Function to normalize the image
def normalize_img(img):
    if isinstance(img, str) or isinstance(img, Path):
        img = nib.load(img)
    data = img.get_fdata()
    data = (data - data.mean()) / data.std()
    return nib.Nifti1Image(data, img.affine)

# List of PMIDs from the image
sample_pmids = [32591929, 
                31114477, 
                31691912, 
                31711031, 
                35874153, 
                30488645]

# Plot parameters
vmax = 6
plot_parameters = {
    "cmap": "bwr",
    "vmax": vmax,
    "views": ["lateral"],
    "colorbar": True,
    "darkness": .3,
    "alpha": 1,
    "bg_on_data": True,
}

# Loop through each PMID and plot with title
for pmid in sample_pmids:
    # Check if pmid exists in groundtruth_test_difumo
    if pmid in groundtruth_test_difumo.index:
        target_image = get_brain_map_by_pmid(kde_groundtruth, pmid)
        groundtruth_sample_voxels_3D = get_voxels_from_difumo__3D_nifti(groundtruth_test_difumo.loc[pmid],target_image)
        # groundtruth_sample_voxels_3D = masker.inverse_transform(groundtruth_sample_voxels_1D)
        # groundtruth_sample_voxels_3D = get_voxels_from_difumo_test(groundtruth_test_difumo.loc[pmid])
    else:
        # If not found in test, use groundtruth_train_difumo
        target_image = get_brain_map_by_pmid(kde_groundtruth, pmid)
        groundtruth_sample_voxels_3D = get_voxels_from_difumo__3D_nifti(groundtruth_train_difumo.loc[pmid],target_image)
        # groundtruth_sample_voxels_3D = get_voxels_from_difumo_test(groundtruth_train_difumo.loc[pmid])
        
    # Plot with title including PMID
    fig = plt.figure(figsize=(8, 6))  # Adjust figure size if needed
    plotting.plot_img_on_surf(
        normalize_img(groundtruth_sample_voxels_3D),
        title=f"PMID: {pmid} - Groundtruth Map",
        **plot_parameters,
    )
    
    # Set the font size for the title
    plt.suptitle(f"PMID: {pmid} - Groundtruth Map", fontsize=24)
    plt.show()


#%% Dice for these sample_pmids

dice_thresholds = [1, 3, 5, 10, 15, 18, 20, 40, 60, 80, 85, 90, 95, 96, 97, 98, 99, 99.5, 99.8]
kde_vs_difumo = []

# for index, el in enumerate(tqdm.tqdm(groundtruth_test_difumo.to_dict("records"))):
# for pmid in tqdm.tqdm(groundtruth_test_difumo.index):
for pmid in tqdm.tqdm(sample_pmids):

    
    groundtruth_test_kde_voxels_3D = get_brain_map_by_pmid(kde_groundtruth, pmid)
    
    if pmid in groundtruth_test_difumo.index:
        # groundtruth_test_difumo_voxels_3D = get_voxels_from_difumo_test(groundtruth_test_difumo.loc[pmid])
        groundtruth_test_difumo_voxels_3D = get_voxels_from_difumo__3D_nifti(groundtruth_test_difumo.loc[pmid], groundtruth_test_kde_voxels_3D)
    else:
        # groundtruth_test_difumo_voxels_3D = get_voxels_from_difumo_test(groundtruth_train_difumo.loc[pmid])
        groundtruth_test_difumo_voxels_3D = get_voxels_from_difumo__3D_nifti(groundtruth_train_difumo.loc[pmid], groundtruth_test_kde_voxels_3D)

    # predicted_test_voxels_3D_resampled = get_voxels_from_difumo__3D_nifti(groundtruth_test_difumo_voxels_3D, groundtruth_test_kde_voxels_3D)
        
    
    kde_vs_difumo_dices = compute_dice(
        groundtruth_test_kde_voxels_3D.get_fdata(), # GT
        groundtruth_test_difumo_voxels_3D.get_fdata(), # Predict
        dice_thresholds,
    )
    print(f"kde_vs_difumo_dices: {kde_vs_difumo_dices}")
    kde_vs_difumo.append(kde_vs_difumo_dices)
    print("kde_vs_difumo: ", np.mean(kde_vs_difumo, axis=0))
    # del predicted_test_voxels_3D_resampled


kde_vs_difumo = np.array(kde_vs_difumo)

kde_vs_difumo_mean = np.mean(kde_vs_difumo, axis=0)
kde_vs_difumo_mean_rounded = np.round(kde_vs_difumo_mean, 2)
print("kde_vs_difumo: ", kde_vs_difumo_mean_rounded)


#%%
# Plot each row of kde_vs_difumo
fs = 24
plt.figure(figsize=(20, 6))
for i in range(kde_vs_difumo.shape[0]):
    plt.plot(dice_thresholds[9:], kde_vs_difumo[i][9:], label=f'PMID {sample_pmids[i]}')

# Add labels, title, and increase font size
plt.xlabel('Brain Map Thresholds', fontsize=fs)
plt.ylabel('Dice Score', fontsize=fs)
plt.title('KDE vs DiFuMo', fontsize=fs)

# Add legend with increased font size
plt.legend(fontsize=fs)

# Increase x-tick and y-tick font size
plt.xticks(dice_thresholds[9:], rotation=90, fontsize=fs)
plt.yticks(fontsize=fs)

# Adjust layout to fit the x-ticks
plt.tight_layout()

# Display the plot
plt.show()


# %%

################################################
###############################################
#%% Dice score between the actual KDE and the KDE reconstructed from DiFuMo

dice_thresholds = [80, 85, 90, 95, 96, 97, 98, 99, 99.5, 99.8]
kde_vs_difumo = []

# for index, el in enumerate(tqdm.tqdm(groundtruth_test_difumo.to_dict("records"))):
for pmid in tqdm.tqdm(groundtruth_test_difumo.index):
    
    groundtruth_test_kde_voxels_3D = get_brain_map_by_pmid(kde_groundtruth, pmid)

    if pmid in groundtruth_test_difumo.index:
        # groundtruth_test_difumo_voxels_3D = get_voxels_from_difumo_test(groundtruth_test_difumo.loc[pmid])
        groundtruth_test_difumo_voxels_3D = get_voxels_from_difumo__3D_nifti(groundtruth_test_difumo.loc[pmid], groundtruth_test_kde_voxels_3D)
    else:
        # groundtruth_test_difumo_voxels_3D = get_voxels_from_difumo_test(groundtruth_train_difumo.loc[pmid])
        groundtruth_test_difumo_voxels_3D = get_voxels_from_difumo__3D_nifti(groundtruth_train_difumo.loc[pmid], groundtruth_test_kde_voxels_3D)

    
    kde_vs_difumo_dices = compute_dice(
        groundtruth_test_kde_voxels_3D.get_fdata(), # GT
        groundtruth_test_difumo_voxels_3D.get_fdata(), # Predict
        dice_thresholds,
    )
    print(f"kde_vs_difumo_dices: {kde_vs_difumo_dices}")
    kde_vs_difumo.append(kde_vs_difumo_dices)
    print("kde_vs_difumo: ", np.mean(kde_vs_difumo, axis=0))
    # del predicted_test_voxels_3D_resampled


kde_vs_difumo = np.array(kde_vs_difumo)

kde_vs_difumo_mean = np.mean(kde_vs_difumo, axis=0)
kde_vs_difumo_mean_rounded = np.round(kde_vs_difumo_mean, 2)
print("kde_vs_difumo: ", kde_vs_difumo_mean_rounded)


save_directory_dice_numpy = '/data/parietal/store3/work/fghayyem/projects/MICCAI_2024/results_Dice_left-out-articles'
numpy_filename_dice = 'Dice_left-out-articles_kde_vs_difumo.npy'

# Ensure the directory exists
os.makedirs(save_directory_dice_numpy, exist_ok=True)

# Full file path for saving
numpy_full_path_dice = os.path.join(save_directory_dice_numpy, numpy_filename_dice)

# Save the kde_vs_difumo object as a NumPy array
np.save(numpy_full_path_dice, kde_vs_difumo)

print(f'kde_vs_difumo saved successfully at {numpy_full_path_dice}')
# %%
