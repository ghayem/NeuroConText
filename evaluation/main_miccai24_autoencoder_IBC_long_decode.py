# %% Load Neuroquery data
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

# Now you can access files and modules in both directories

from collections import defaultdict
from functools import partial
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from scripts.downstream_task.clip.clip.layers import ClipModel, ClipModel_autoencoder, MLP, ProjectionHead, ResidualHead, ProjectionHead_decoder, ResidualHead_decoder, ResidualHead_autoencoder
from scripts.downstream_task.clip.clip.losses import ClipLoss, SigLipLoss, AdMClipLoss
from scripts.downstream_task.clip.clip.plotting import plot_matrix
from scripts.downstream_task.clip.clip.training import (
    check_model_parameter_callback, count_parameters,
    diagonal_callback, non_diagonal_callback,
    predict, predict_autoencoder, recall_n_callback, single_input_predict, single_input_train, train, train_autoencoder,
)
from scripts.downstream_task.clip.utils import load_publications_data , load_atlas_masked, extract_difumo_from_images
from sklearn.preprocessing import Normalizer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.metrics import mix_match
from src.utils import plot_training, recall_n
from src.nnod import preprocessing_on_samples
from neuroquery.img_utils import get_masker

import transformers
from src.constants import CACHE_PATH
from src.embeddings import batch_embed_texts_by_chunks, chunk_tokenize_texts, embed

# %%

# df_name = "abstract" # you should set either you want to work with title, abstract, or body

from experiment_setting import df_name, difumo_dimension, llm_key, train_size_control, flag_train_size_control

#%%

# Paths and settings
base_path = '/data/parietal/store3/work/fghayyem/projects/MICCAI_2024/results'

if flag_train_size_control==True:
    test_data_folder_name = f'text2brain_clip_comparison_train_size_{train_size_control}'
else:
    test_data_folder_name = f'text2brain_clip_comparison'

# %%
(
    pubget_text_data,
    pubget_metadata,
    kde_brain_maps,
    train_pmids,
    test_pmids,
    train_gaussian_difumo,
    test_gaussian_difumo,
    train_title_embeddings,
    test_title_embeddings,
    train_abstract_embeddings,
    test_abstract_embeddings,
    # train_keywords_embeddings,
    # test_keywords_embeddings,
    train_body_embeddings,
    test_body_embeddings,
    train_body_embeddings_chunks,
    test_body_embeddings_chunks,
    body_embeddings_chunks,
) = load_publications_data(
    dimension=difumo_dimension,
    test_size=1000,
    train_size_control = train_size_control,
    flag_train_size_control=flag_train_size_control,
    key=llm_key,  # possible keys are {"gpt-neo-125m", "gpt-neo-125m-finetuned", "gpt-neo-1.3B"}
)


#%% 
# If you want to use all the body chunks to apply text augmentation
# set flag_text_augment to True

flag_text_augment = False

if flag_text_augment:

    train_body_embeddings_backup = train_body_embeddings
    test_body_embeddings_backup = test_body_embeddings
    train_gaussian_difumo_backup = train_gaussian_difumo
    test_gaussian_difumo_backup = test_gaussian_difumo

    train_body_embeddings = train_body_embeddings_chunks
    test_body_embeddings = test_body_embeddings_chunks

    train_gaussian_difumo_reset = train_gaussian_difumo.reset_index()
    train_gaussian_difumo_reset.rename(columns={'index': 'pmid'}, inplace=True)
    pmid_to_row_map = train_gaussian_difumo_reset.set_index('pmid').T.to_dict('list')
    train_body_embeddings_chunks_reset = train_body_embeddings_chunks.reset_index()
    rows_to_append = [[pmid] + pmid_to_row_map[pmid] for pmid in train_body_embeddings_chunks_reset['pmid']]
    replicated_train_gaussian_difumo = pd.DataFrame(rows_to_append, columns=train_gaussian_difumo_reset.columns)
    replicated_train_gaussian_difumo.set_index('pmid', inplace=True)

    test_gaussian_difumo_reset = test_gaussian_difumo.reset_index()
    test_gaussian_difumo_reset.rename(columns={'index': 'pmid'}, inplace=True)
    pmid_to_row_map = test_gaussian_difumo_reset.set_index('pmid').T.to_dict('list')
    test_body_embeddings_chunks_reset = test_body_embeddings_chunks.reset_index()
    rows_to_append = [[pmid] + pmid_to_row_map[pmid] for pmid in test_body_embeddings_chunks_reset['pmid']]
    replicated_test_gaussian_difumo = pd.DataFrame(rows_to_append, columns=test_gaussian_difumo_reset.columns)
    replicated_test_gaussian_difumo.set_index('pmid', inplace=True)

    train_gaussian_difumo = replicated_train_gaussian_difumo
    test_gaussian_difumo = replicated_test_gaussian_difumo

# train_body_embeddings = train_body_embeddings_backup
# test_body_embeddings = test_body_embeddings_backup
# train_gaussian_difumo = train_gaussian_difumo_backup
# test_gaussian_difumo = test_gaussian_difumo_backup

#%% Add pmid to pubget_text_data

pubget_text_data['pmcid'] = pubget_text_data['pmcid'].astype(int)


pubget_metadata['pmcid'] = pubget_metadata['pmcid'].astype(int)

# Now perform the merge
pubget_text_data_with_pmid = pd.merge(pubget_text_data, pubget_metadata[['pmcid', 'pmid']], on='pmcid', how='left')

# check
pubget_metadata1 = pubget_metadata
pubget_metadata1['pmcid'] = pubget_metadata1['pmcid'].astype(str)
desired_pmid = pubget_metadata1.loc[pubget_metadata1['pmcid'] == '3591410', 'pmid']

#%%
test_pmids = set(test_pmids)

pubget_text_data_unique = pubget_text_data_with_pmid.drop_duplicates(subset='pmid')
pubget_text_data_test = pubget_text_data_unique[pubget_text_data_unique['pmid'].isin(test_pmids)]
test_pubget_pmids = pubget_text_data_test["pmid"]

#####
neuroquery_text_data = pd.read_csv("/data/parietal/store2/data/neuroquery/preprocessing/flat_corpus/split_corpus_2017-10-05T19-44-16_labelled_documents.csv")
neuroquery_text_data_unique = neuroquery_text_data.drop_duplicates(subset='pmid')
nq_pmids = set(neuroquery_text_data["pmid"])
test_nq_pmids = nq_pmids.intersection(test_pmids)
test_shared_nq_pubget_pmids = set(test_nq_pmids).intersection(set(test_pubget_pmids))
test_unique_nq_pmids = test_nq_pmids - test_shared_nq_pubget_pmids
test_unique_nq_pmids = test_unique_nq_pmids
neuroquery_text_data_test = neuroquery_text_data_unique[neuroquery_text_data['pmid'].isin(test_unique_nq_pmids)]

title_test = pd.concat([pubget_text_data_test[["title","pmid"]], neuroquery_text_data_test[["title","pmid"]]])
keywords_test = pd.concat([pubget_text_data_test[["keywords","pmid"]], neuroquery_text_data_test[["keywords","pmid"]]])
abstract_test = pd.concat([pubget_text_data_test[["abstract","pmid"]], neuroquery_text_data_test[["abstract","pmid"]]])
body_test = pd.concat([pubget_text_data_test[["body","pmid"]], neuroquery_text_data_test[["body","pmid"]]])

title_test['pmid'] = title_test['pmid'].astype(int)
keywords_test['pmid'] = keywords_test['pmid'].astype(int)
abstract_test['pmid'] = abstract_test['pmid'].astype(int)
body_test['pmid'] = body_test['pmid'].astype(int)

# %%
# Those concatenations are here because at some point I tried to used each different
# embeddings (title, abstract, keywords, body). It did not work well, so I only kept the body
# but I left the concatenation in case I want to try again.
# train_text_embeddings = np.concatenate([
#     # train_title_embeddings,
#     # train_keywords_embeddings,
#     # train_abstract_embeddings,
#     train_body_embeddings,
# ], axis=0)

# test_text_embeddings = np.concatenate([
#     # test_title_embeddings,
#     # test_keywords_embeddings,
#     # test_abstract_embeddings,
#     test_body_embeddings,
# ], axis=0)

# Dictionary mapping df_name values to their corresponding embeddings
train_embeddings_dict = {
    "title": train_title_embeddings,
    # "keywords": train_keywords_embeddings,
    "abstract": train_abstract_embeddings,
    "body": train_body_embeddings,
}

test_embeddings_dict = {
    "title": test_title_embeddings,
    # "keywords": test_keywords_embeddings,
    "abstract": test_abstract_embeddings,
    "body": test_body_embeddings,
}

# Dynamically select the embeddings to concatenate based on df_name
train_text_embeddings = np.concatenate([
    train_embeddings_dict[df_name],
], axis=0)

test_text_embeddings = np.concatenate([
    test_embeddings_dict[df_name],
], axis=0)

train_gaussian_embeddings = np.concatenate([
    # train_gaussian_difumo,
    # train_gaussian_difumo,
    train_gaussian_difumo,
], axis=0)
test_gaussian_embeddings = np.concatenate([
    # test_gaussian_difumo,
    # test_gaussian_difumo,
    test_gaussian_difumo,
], axis=0)

# %%
preprocess_text, preprocess_gaussian = True, False

preprocessed_train_text_embeddings = train_text_embeddings
preprocessed_test_text_embeddings = test_text_embeddings
if preprocess_text:
    text_preprocessing_pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        # ("normalizer", Normalizer()),
        # ('pca', PCA(n_components=0.95)),
    ])
    preprocessed_train_text_embeddings = text_preprocessing_pipeline.fit_transform(
        preprocessed_train_text_embeddings,
    )
    preprocessed_test_text_embeddings = text_preprocessing_pipeline.transform(
        preprocessed_test_text_embeddings,
    )

preprocessed_train_gaussian_embeddings = train_gaussian_embeddings
preprocessed_test_gaussian_embeddings = test_gaussian_embeddings
if preprocess_gaussian:
    gaussian_preprocessing_pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True))
        # ("normalizer", Normalizer()),
        # ('pca', PCA(n_components=0.95)),
    ])
    preprocessed_train_gaussian_embeddings = gaussian_preprocessing_pipeline.fit_transform(
        preprocessed_train_gaussian_embeddings,
    )
    preprocessed_test_gaussian_embeddings = gaussian_preprocessing_pipeline.transform(
        preprocessed_test_gaussian_embeddings,
    )
print(f"Text features: {preprocessed_train_text_embeddings.shape[1]}")
print(f"Gaussian features: {preprocessed_train_gaussian_embeddings.shape[1]}")


#%% save train test text data and test groundtruth DiFuMo coefficients in the "Data_articles_CurrentExperiment_NeuroConText_AE" directory

data_current_exp_directory = '/data/parietal/store3/work/fghayyem/projects/MICCAI_2024/Data_articles_CurrentExperiment_NeuroConText_AE'
os.makedirs(data_current_exp_directory, exist_ok=True)

title_test.to_csv(os.path.join(data_current_exp_directory, 'title_test.csv'), index=False)
keywords_test.to_csv(os.path.join(data_current_exp_directory, 'keywords_test.csv'), index=False)
abstract_test.to_csv(os.path.join(data_current_exp_directory, 'abstract_test.csv'), index=False)
body_test.to_csv(os.path.join(data_current_exp_directory, 'body_test.csv'), index=False)

pd.DataFrame(data=preprocessing_on_samples(preprocessed_test_gaussian_embeddings), index=test_body_embeddings.index).to_csv(os.path.join(data_current_exp_directory, "preprocessed_test_gaussian_embeddings_groundtruth_difumo.csv"))


#%%
################################################################################
######################## Train the models Ensumble ########################
################################## Direct ##################################

######### Define criterion #########

# Define criterion_contrastive and criterion_mse
criterion_contrastive = ClipLoss()
criterion_mse = nn.MSELoss()
criterion = ClipLoss()

# Check if the criterion is ClipLoss
is_clip_loss = criterion.__class__ == ClipLoss
loss_specific_kwargs = {
    "logit_scale": 10 if is_clip_loss else np.log(10),
    "logit_bias": None if is_clip_loss else -10,
}

######### Set up parameters #########
validation_size = 1000
batch_size = 125
lr = 1e-4
lr_encoder = 1e-4
lr_decoder = 1e-4 #1e-2
weight_decay = 0.1 # 0.1
dropout = 0.6
dropout_decoder = 0.6
num_epochs = 50
output_size = difumo_dimension
decoder_hidden_size = difumo_dimension
device = 'cuda' if torch.cuda.is_available() else 'cpu'
k_fold = KFold(n_splits=5)
number_of_folds_to_run = 1
# beta_grid = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0]
beta = 0e-0 # 1e-4 # MSE loss hyperparameter
alpha = 1e-0 # 1ontrastive loss hyperparameter
flag_freez_index = False

plot_verbose = True

## set the test data
test_dataset = TensorDataset(
    torch.from_numpy(preprocessing_on_samples(preprocessed_test_gaussian_embeddings)).float(),
    torch.from_numpy(preprocessed_test_text_embeddings).float(),
)


test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# %%
print(f"Using device: {device}")
validation_size = 1000
k_fold = KFold(n_splits=len(preprocessed_train_text_embeddings) // validation_size)

recall_fn = partial(recall_n, thresh=0.95, reduce_mean=True)

output_dir = Path(__file__).parent / "main_output"
output_dir.mkdir(exist_ok=True)


metrics = {
    "train": defaultdict(list),
    "validation": defaultdict(list),
    "test": defaultdict(list),
}

if beta == 0 and flag_freez_index:
    print("we freezed the train and the validation indeces")
    train_index_backup = train_index
    val_index_backup = val_index

number_of_folds_to_run = 1
for fold, (train_index, val_index) in enumerate(k_fold.split(preprocessed_train_text_embeddings)):
    val_index = val_index[:validation_size]  # Strict 1000 validation samples
    if fold >= number_of_folds_to_run:
        break

    if beta == 0 and flag_freez_index:
        train_index = train_index_backup
        val_index = val_index_backup

    train_dataset = TensorDataset(
        torch.from_numpy(preprocessing_on_samples(preprocessed_train_gaussian_embeddings[train_index])).float(),
        torch.from_numpy(preprocessed_train_text_embeddings[train_index]).float(),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(
        torch.from_numpy(preprocessing_on_samples(preprocessed_train_gaussian_embeddings[val_index])).float(),
        torch.from_numpy(preprocessed_train_text_embeddings[val_index]).float(),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ClipModel_autoencoder(
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
        decoder_model = nn.Sequential(
            ResidualHead(output_size, dropout=dropout),
            ResidualHead(output_size, dropout=dropout),
            # ResidualHead_decoder(output_size, dropout=dropout),
            # ResidualHead_decoder(output_size, dropout=dropout_decoder),
            # ResidualHead(output_size, dropout=dropout),
            # ResidualHead(output_size, dropout=dropout),
            # ResidualHead_decoder(output_size, dropout=dropout),
        ),
        **loss_specific_kwargs,
    )
    model.train()

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
    scheduler = None  # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*num_epochs)
    output_dir = Path(__file__).parent / "main_output"
    output_dir.mkdir(exist_ok=True)

    clip_model, clip_train_loss, clip_val_loss, loss_contrastive_train, loss_contrastive_val, loss_mse_train, loss_mse_val, callback_outputs = train_autoencoder(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_encoder=optimizer_encoder,
        optimizer_decoder=optimizer_decoder,
        scheduler=scheduler,
        criterion=criterion,
        beta = beta,
        alpha = alpha,
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

    clip_model.eval()

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

    if plot_verbose:
        callback_plot_kwargs = [
            {"ylabel": "Validation\nRecall@10", "color": "b", "ylim": [0, 1]},
            {"ylabel": "Diagonal Mean", "color": "b", "ylim": [1e-7, 1], "yscale": "log"},
            {"ylabel": "Non-diagonal Mean", "color": "b", "ylim": [1e-7, 1], "yscale": "log"},
            {"ylabel": "Logit scale", "color": "black"},
            {"ylabel": "Logit bias", "color": "black"},
        ]
        print("Loss Contrastive")
        plot_training(
            loss_contrastive_train,
            loss_contrastive_val,
            callback_outputs,
            callback_kwargs=callback_plot_kwargs,
        )

    if plot_verbose:
        callback_plot_kwargs = [
            {"ylabel": "Validation\nRecall@10", "color": "b", "ylim": [0, 1]},
            {"ylabel": "Diagonal Mean", "color": "b", "ylim": [1e-7, 1], "yscale": "log"},
            {"ylabel": "Non-diagonal Mean", "color": "b", "ylim": [1e-7, 1], "yscale": "log"},
            {"ylabel": "Logit scale", "color": "black"},
            {"ylabel": "Logit bias", "color": "black"},
        ]
        print("Loss MSE")
        plot_training(
            loss_mse_train,
            loss_mse_val,
            callback_outputs,
            callback_kwargs=callback_plot_kwargs,
        )

    # Define a small train dataset to get metrics faster
    small_train_dataset = TensorDataset(
        torch.from_numpy(preprocessing_on_samples(preprocessed_train_gaussian_embeddings[train_index])[:1000]).float(),
        torch.from_numpy(preprocessed_train_text_embeddings[train_index][:1000]).float(),
    )
    small_train_loader = DataLoader(small_train_dataset, batch_size=batch_size, shuffle=True)
    for loader_name, loader, weights_path in [
        ("train", small_train_loader, output_dir / "last.pt"),
        ("validation", val_loader, output_dir / "best_val.pt"),
        ("test", test_loader, output_dir / "best_val.pt"),
    ]:
        # clip_model.load_state_dict(torch.load(weights_path))

        image_embeddings, text_embeddings, latent_decoded = predict_autoencoder(clip_model, loader, device=device)
        n_iterations = 20  # Number of random selections
        subset_size = 1000  # Size of each random subset
        recall_at_10_results = []
        recall_at_100_results = []
        mix_match_results = []

        for iteration in range(n_iterations):
            # Random selection of indices for text and image embeddings
            indices = np.random.choice(len(text_embeddings), size=subset_size, replace=False)
            
            # Subsetting the embeddings
            text_embeddings_subset = text_embeddings[indices]
            image_embeddings_subset = image_embeddings[indices]
            
            # Calculating similarity for the subset
            similarity_subset = (text_embeddings_subset @ image_embeddings_subset.T).softmax(dim=1).numpy()
            
            if iteration == 0 and plot_verbose:
                # Plot similarity matrices for the first subset only
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
                gauss_similarity_subset = (image_embeddings_subset @ image_embeddings_subset.T).numpy()
                plot_matrix(gauss_similarity_subset[:100, :100], ax=axes[0], title="Gauss-to-Gauss Subset")
                text_similarity_subset = (text_embeddings_subset @ text_embeddings_subset.T).numpy()
                plot_matrix(text_similarity_subset[:100, :100], ax=axes[1], title="Text-to-Text Subset")
                plot_matrix(similarity_subset[:100, :100], ax=axes[2], title="Gauss-to-Text Subset")
                fig.suptitle(f"Learnt similarities - {loader_name} Subset")
                plt.tight_layout()
                plt.show()
            
            # Calculating recalls and mix_match for the subset
            recall_at_10 = recall_fn(similarity_subset, np.eye(len(similarity_subset)), n_first=10)
            recall_at_100 = recall_fn(similarity_subset, np.eye(len(similarity_subset)), n_first=100)
            mix_match_score = 100 * mix_match(similarity_subset)
            
            # Storing results for each iteration
            recall_at_10_results.append(recall_at_10)
            recall_at_100_results.append(recall_at_100)
            mix_match_results.append(mix_match_score)

        # Calculating mean and std for the metrics
        recall_at_10_avg = 100 * np.mean(recall_at_10_results)
        recall_at_10_std = 100 * np.std(recall_at_10_results)
        recall_at_100_avg = 100 * np.mean(recall_at_100_results)
        recall_at_100_std = 100 * np.std(recall_at_100_results)
        mix_match_avg = np.mean(mix_match_results)
        mix_match_std = np.std(mix_match_results)

        # Updating the metrics dictionary for the loader_name with average and std
        metrics[loader_name]["recall@10_avg"].append(recall_at_10_avg)
        metrics[loader_name]["recall@10_std"].append(recall_at_10_std)
        metrics[loader_name]["recall@100_avg"].append(recall_at_100_avg)
        metrics[loader_name]["recall@100_std"].append(recall_at_100_std)
        metrics[loader_name]["mix_match_avg"].append(mix_match_avg)
        metrics[loader_name]["mix_match_std"].append(mix_match_std)

        print(f"fold: {fold}")

print(f"Metrics after processing")
for loader_name in ["train", "validation", "test"]:
    print("="*10, loader_name, "="*10)
    for metric_name in ["recall@10", "recall@100", "mix_match"]:
        avg_metric_name = f"{metric_name}_avg"
        std_metric_name = f"{metric_name}_std"
        if avg_metric_name in metrics[loader_name] and std_metric_name in metrics[loader_name]:
            avg_value = np.mean(metrics[loader_name][avg_metric_name])
            std_value = np.mean(metrics[loader_name][std_metric_name])  # Assuming you want to average the std across folds if applicable
            print(f"{metric_name}: {avg_value:.3f} +- {std_value:.3f}")
        else:
            print(f"{metric_name} not available for {loader_name}")



print("############## mean and std #################")

for loader_name in ["train", "validation", "test"]:
    if loader_name not in metrics:
        print(f"No data for {loader_name}")
        continue
    
    print("="*10, loader_name, "="*10)
    for metric_name in ["recall@10", "recall@100", "mix_match"]:
        avg_metric_name = f"{metric_name}_avg"
        std_metric_name = f"{metric_name}_std"
        if avg_metric_name in metrics[loader_name] and std_metric_name in metrics[loader_name]:
            avg_values = metrics[loader_name][avg_metric_name]
            std_values = metrics[loader_name][std_metric_name]
            
            mean_avg_value = np.mean(avg_values)
            std_avg_value = np.std(avg_values)  # Calculate std of the average values
            
            print(f"{metric_name}: {mean_avg_value:.3f} ± {std_avg_value:.3f}")
        else:
            print(f"{metric_name} not available for {loader_name}")

print(f"##### Parameters #####")
print(f"dropout: {dropout}")
print(f"num_epochs: {num_epochs}")
print(f"beta: {beta}")
print(f"alpha: {alpha}")
print(f"lr_encoder: {lr_encoder}")
print(f"lr_decoder: {lr_decoder}")







#%%
##########################################################
######## Decoding from text latent to Brain maps #########

# clip_model.load_state_dict(torch.load(output_dir / "best_val.pt"))

train_dataset = TensorDataset(
    torch.from_numpy(preprocessing_on_samples(preprocessed_train_gaussian_embeddings)[train_index]).float(),
    torch.from_numpy(preprocessed_train_text_embeddings[train_index]).float(),
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
train_image_embeddings, train_text_embeddings, _ = predict_autoencoder(clip_model, train_loader, device=device)
train_text_embeddings = train_text_embeddings.cpu().numpy()
train_image_embeddings = train_image_embeddings.cpu().numpy()

val_dataset = TensorDataset(
    torch.from_numpy(preprocessing_on_samples(preprocessed_train_gaussian_embeddings)[val_index]).float(),
    torch.from_numpy(preprocessed_train_text_embeddings[val_index]).float(),
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
val_image_embeddings, val_text_embeddings, _= predict_autoencoder(clip_model, val_loader, device=device)
val_text_embeddings = val_text_embeddings.cpu().numpy()
val_image_embeddings = val_image_embeddings.cpu().numpy()


test_dataset = TensorDataset(
    torch.from_numpy(preprocessing_on_samples(preprocessed_test_gaussian_embeddings)).float(),
    torch.from_numpy(preprocessed_test_text_embeddings).float(),
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_image_embeddings, test_text_embeddings, _ = predict_autoencoder(clip_model, test_loader, device=device)
test_text_embeddings = test_text_embeddings.cpu().numpy()
test_image_embeddings = test_image_embeddings.cpu().numpy()


# %%
# print("Load atlases")
# masker, atlas, atlas_masked, mask = load_atlas_masked(512)
# nq_masker = get_masker(mask_img=None, target_affine=(4, 4, 4))

# %%

reconstruction_dir = Path(__file__).parent / "reconstruction_output"
reconstruction_dir.mkdir(exist_ok=True, parents=True)

###################################
#%% reconstruct from test data

( test_clip_AE_image_embeddings, 
 test_clip_AE_text_embeddings, 
 test_clip_AE_latent_decoded
) = predict_autoencoder(clip_model, test_loader, device=device)

print("Clip embeddings done.")
test_clip_AE_text_embeddings = test_clip_AE_text_embeddings.cpu().numpy()
test_clip_AE_latent_decoded = test_clip_AE_latent_decoded.numpy()

pd.DataFrame(data=test_clip_AE_latent_decoded, index=test_body_embeddings.index).to_csv(reconstruction_dir / "test_clip_AE_latent_decoded_difumo.csv")
pd.DataFrame(data=preprocessing_on_samples(preprocessed_test_gaussian_embeddings), index=test_body_embeddings.index).to_csv(reconstruction_dir / "preprocessed_test_gaussian_embeddings_groundtruth_difumo.csv")


print(f"reconstruction_dir: {reconstruction_dir}")

###################################
#%% reconstruct from train data

from torch.utils.data import TensorDataset, DataLoader
import random

( train_clip_AE_image_embeddings, 
train_clip_AE_text_embeddings, 
train_clip_AE_latent_decoded 
) = predict_autoencoder(clip_model, train_loader, device=device)

print("Clip embeddings done.")
train_clip_AE_text_embeddings = train_clip_AE_text_embeddings.cpu().numpy()
train_clip_AE_latent_decoded = train_clip_AE_latent_decoded.numpy()

pd.DataFrame(data=train_clip_AE_latent_decoded, index=train_body_embeddings.index[train_index]).to_csv(reconstruction_dir / "train_clip_AE_latent_decoded_difumo.csv")
pd.DataFrame(data=preprocessing_on_samples(preprocessed_train_gaussian_embeddings[train_index]), index=train_body_embeddings.index[train_index]).to_csv(reconstruction_dir / "preprocessed_train_gaussian_embeddings_groundtruth_difumo.csv")

pd.DataFrame(data=preprocessing_on_samples(preprocessed_train_gaussian_embeddings[val_index]), index=train_body_embeddings.index[val_index]).to_csv(reconstruction_dir / "preprocessed_val_gaussian_embeddings_groundtruth_difumo.csv")

print(f"reconstruction_dir: {reconstruction_dir}")


# %% Reconstrucion with original (org) NeuroConTet paper: separately train the decoder
#### you should set beta=0
#### you should have a new data loader based on the new latents


reconstruction_train_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(train_text_embeddings).float(),
        # torch.from_numpy(train_image_embeddings).float(),
        torch.from_numpy(preprocessing_on_samples(preprocessed_train_gaussian_embeddings)[train_index]).float(),
    ),
    batch_size=256,
    shuffle=True,
)

reconstruction_val_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(val_text_embeddings).float(),
        # torch.from_numpy(val_image_embeddings).float(),
        torch.from_numpy(preprocessing_on_samples(preprocessed_train_gaussian_embeddings)[val_index]).float(),
    ),
    batch_size=256,
    shuffle=True,
)

reconstruction_test_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(test_text_embeddings).float(),
        # torch.from_numpy(test_image_embeddings).float(),
        torch.from_numpy(preprocessing_on_samples(preprocessed_test_gaussian_embeddings)).float(),
    ),
    batch_size=256,
    shuffle=False,
)

reconstruction_model = nn.Sequential(
    ResidualHead(output_size, dropout=dropout),
    # ResidualHead(output_size, dropout=dropout),
    ResidualHead_decoder(output_size, dropout=dropout),
)

reconstruction_model.train()

reconstruction_dir = Path(__file__).parent / "reconstruction_output"
reconstruction_dir.mkdir(exist_ok=True, parents=True)
reconstruction_model, reconstruction_train_loss, reconstruction_val_loss = single_input_train(
    reconstruction_model,
    train_loader=reconstruction_train_loader,
    val_loader=reconstruction_val_loader,
    optimizer=torch.optim.AdamW(    
        reconstruction_model.parameters(),
        lr=1e-4,
        weight_decay=weight_decay,
    ),
    scheduler=None,
    criterion=nn.MSELoss(),
    num_epochs=50,
    device=device,
    verbose=True,
    output_dir=reconstruction_dir,
)
# reconstruction_model.load_state_dict(torch.load(reconstruction_dir / "best_val.pt"))
reconstruction_model.eval()

print("Plot training losses")
plt.plot(reconstruction_train_loss, label="train")
plt.plot(reconstruction_val_loss, label="val")
plt.legend()
plt.show()


# Prediction for test set
test_clip_org_latent_decoded = single_input_predict(reconstruction_model, reconstruction_test_loader, device=device)

test_clip_org_latent_decoded = test_clip_org_latent_decoded.numpy()
pd.DataFrame(data=test_clip_org_latent_decoded, index=test_body_embeddings.index).to_csv(reconstruction_dir / "test_clip_org_latent_decoded_difumo.csv")
pd.DataFrame(data=test_clip_org_latent_decoded, index=test_body_embeddings.index).to_csv(reconstruction_dir / "test_clip_org_latent_decoded_difumo_50Epochs.csv")

# Prediction for train set
train_clip_org_latent_decoded = single_input_predict(reconstruction_model, reconstruction_train_loader, device=device)

train_clip_org_latent_decoded = train_clip_org_latent_decoded.numpy()
pd.DataFrame(data=train_clip_org_latent_decoded, index=train_body_embeddings.index[train_index]).to_csv(reconstruction_dir / "train_clip_org_latent_decoded_difumo.csv")
pd.DataFrame(data=train_clip_org_latent_decoded, index=train_body_embeddings.index[train_index]).to_csv(reconstruction_dir / "train_clip_org_latent_decoded_difumo_50Epochs.csv")


# Prediction for validation set
val_clip_org_latent_decoded = single_input_predict(reconstruction_model, reconstruction_val_loader, device=device)

val_clip_org_latent_decoded = val_clip_org_latent_decoded.numpy()
pd.DataFrame(data=val_clip_org_latent_decoded, index=train_body_embeddings.index[val_index]).to_csv(reconstruction_dir / "val_clip_org_latent_decoded_difumo.csv")

######################################
# %% IBC contrast reconstruction 

## By LONG we mean that we have added a new column to the metadata where the discription are 
## extended by using chat-gpt4.
# %%
### load contrast_embeddings and contrast_ids
ibc_MistralEmbed_folder = Path(__file__).parent / "baselines" / "ibc_MistralEmbed" / "ContrastDefinition_long"

contrast_embeddings = np.load(os.path.join(ibc_MistralEmbed_folder, "contrast_embeddings.npy"))
contrast_ids = np.load(os.path.join(ibc_MistralEmbed_folder, "contrast_ids.npy"))

#%%
### get the difumo of the IBC contrasts
contrast_images_path = '/data/parietal/store3/work/fghayyem/projects/MICCAI_2024/retreat-neuro-llm/scripts/downstream_task/clip/clip/baselines/ibc_average'
contrast_difumo_path = f'/data/parietal/store3/work/fghayyem/projects/MICCAI_2024/retreat-neuro-llm/scripts/downstream_task/clip/clip/baselines/ibc_average_difumo/ibc_contrast_gaussian_difumo_{difumo_dimension}.csv'
contrast_gaussian_difumo_all = extract_difumo_from_images(
    contrast_images_path,
    contrast_difumo_path,
    dimension=difumo_dimension,
    target_affine=(6, 6, 6),
    fwhm=9.0,
    normalize=True,
)

contrast_gaussian_difumo = contrast_gaussian_difumo_all.loc[pd.Index(contrast_ids).drop_duplicates()]

# Concatenate data for Gaussian embeddings
contrast_gaussian_embeddings = np.concatenate([
    contrast_gaussian_difumo,
], axis=0)

preprocessed_contrast_gaussian_embeddings = contrast_gaussian_embeddings

if preprocess_gaussian:
    gaussian_preprocessing_pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True))
        # Add other steps like Normalizer() or PCA() if needed
    ])
    preprocessed_train_gaussian_embeddings = gaussian_preprocessing_pipeline.fit_transform(
        preprocessed_train_gaussian_embeddings,
    )
    # Fit and transform the train embeddings, then transform the contrast embeddings
    preprocessed_contrast_gaussian_embeddings = gaussian_preprocessing_pipeline.transform(preprocessed_contrast_gaussian_embeddings)

print(f" contrast Gaussian features: {preprocessed_contrast_gaussian_embeddings.shape[1]}")

# Save preprocessed embeddings to CSV
contrast_preprocessed_output_path = f'/data/parietal/store3/work/fghayyem/projects/MICCAI_2024/retreat-neuro-llm/scripts/downstream_task/clip/clip/baselines/ibc_average_difumo/preprocessed_contrast_gaussian_embeddings_groundtruth_difumo.csv'
pd.DataFrame(data=preprocessing_on_samples(preprocessed_contrast_gaussian_embeddings), index=pd.Index(contrast_ids).drop_duplicates()).to_csv(contrast_preprocessed_output_path)
#%% if necessary to average the LLM embeddings of chunks for an article with several chunks

from collections import defaultdict, Counter
import unicodedata

# Normalize and strip whitespace from the elements in contrast_ids
normalized_contrast_ids = np.array([unicodedata.normalize('NFC', x).strip() for x in contrast_ids])

# Find duplicates and calculate their averages
id_to_embeddings = defaultdict(list)

# Populate the dictionary with embeddings for each id
for idx, uid in enumerate(normalized_contrast_ids):
    id_to_embeddings[uid].append(contrast_embeddings[idx])

# Prepare new lists for unique ids and their corresponding embeddings
new_contrast_ids = []
new_contrast_embeddings = []

for uid, embeddings in id_to_embeddings.items():
    if len(embeddings) > 1:
        # Calculate the average for duplicate embeddings
        avg_embedding = np.mean(embeddings, axis=0)
    else:
        # Use the original embedding if there's only one
        avg_embedding = embeddings[0]
    
    new_contrast_ids.append(uid)
    new_contrast_embeddings.append(avg_embedding)

# Convert lists back to numpy arrays
new_contrast_ids = np.array(new_contrast_ids)
new_contrast_embeddings = np.array(new_contrast_embeddings)

# Verify the new shape
print("New shape of contrast_ids:", new_contrast_ids.shape)
print("New shape of contrast_embeddings:", new_contrast_embeddings.shape)

print(f"contrast_embeddings shape before: {contrast_embeddings.shape}")

contrast_ids = new_contrast_ids
contrast_embeddings = new_contrast_embeddings

print(f"contrast_embeddings shape after: {contrast_embeddings.shape}")
#%%

if preprocess_text:
    text_preprocessing_pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        # ("normalizer", Normalizer()),
        # ('pca', PCA(n_components=0.95)),
    ])
   
    preprocessed_train_text_embeddings = text_preprocessing_pipeline.fit_transform(
        preprocessed_train_text_embeddings,
    )

    contrast_embeddings = text_preprocessing_pipeline.transform(
        contrast_embeddings,
    )

contrast_dataset = TensorDataset(
    torch.from_numpy(preprocessing_on_samples(preprocessed_contrast_gaussian_embeddings)).float(),
    torch.from_numpy(preprocessing_on_samples(contrast_embeddings)).float(),
)
# contrast_dataset = TensorDataset(
#     torch.from_numpy(np.random.random((contrast_embeddings.shape[0], 512))).float(),
#     torch.from_numpy(contrast_embeddings).float(),
# )
contrast_loader = DataLoader(contrast_dataset, batch_size=batch_size, shuffle=False)

# _, contrast_clip_embeddings = predict(clip_model, contrast_loader, device=device)
# print("Clip embeddings done.")

ibc_image_embeddings, ibc_text_embeddings, ibc_text_decode = predict_autoencoder(clip_model, contrast_loader, device=device)
contrast_clip_embeddings = ibc_text_embeddings

contrast_clip_embeddings = contrast_clip_embeddings.cpu().numpy()
contrast_img_clip_embeddings = ibc_image_embeddings.cpu().numpy()

#%% 
# Calculate the association metrics for IBC description

contrast_text_image_similarity = contrast_clip_embeddings @ contrast_img_clip_embeddings.T

# Set up the figure and single axis
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the matrix
plot_matrix(contrast_text_image_similarity, ax=ax, title="Contrast Text-Image Similarity")

# Add a title and display the plot
fig.suptitle("Learnt Similarities - IBC Contrast")
plt.tight_layout()
plt.show()

recall_at_10 = 100 * recall_fn(contrast_text_image_similarity, np.eye(len(contrast_text_image_similarity)), n_first=10)
recall_at_100 = 100 * recall_fn(contrast_text_image_similarity, np.eye(len(contrast_text_image_similarity)), n_first=100)
mix_match_score = 100 * mix_match(contrast_text_image_similarity)

# Print the results
print(f"Recall@10: {recall_at_10:.2f}%")
print(f"Recall@100: {recall_at_100:.2f}%")
print(f"Mix-Match Score: {mix_match_score:.2f}%")

#%%
contrast_reconstructed_loader = DataLoader(
    TensorDataset(torch.from_numpy(contrast_clip_embeddings).float()),
    batch_size=256,
    shuffle=False,
)
contrast_predictions = single_input_predict(reconstruction_model, contrast_reconstructed_loader, device=device)

# pd.DataFrame(data=contrast_predictions.numpy(), index=average_ibc_metadata["id"]).to_csv(reconstruction_dir / "average_ibc_difumo.csv")

#%%
#### If you want to use the original NeuroConText_AE decoder for reconstruction, uncomment below
df = pd.DataFrame(data=contrast_predictions.numpy(), index=contrast_ids)
df.index.name = "id"
# df.to_csv(reconstruction_dir / "average_ibc_difumo.csv")
df.to_csv(reconstruction_dir / "average_ibc_difumo_org_short.csv")

#%%
#### If you want to use the decoder of NeuroConText_AE for reconstruction, uncomment below
df = pd.DataFrame(data=ibc_text_decode.numpy(), index=contrast_ids)
df.index.name = "id"
# df.to_csv(reconstruction_dir / "average_ibc_difumo.csv")
df.to_csv(reconstruction_dir / "average_ibc_difumo_AE_long.csv")


# %%
