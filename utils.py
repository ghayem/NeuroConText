import joblib
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from functools import partial
from joblib import delayed
from loguru import logger
from pathlib import Path

from neuroquery.img_utils import get_masker, iter_coordinates_to_maps
from neuroquery import datasets as nq_datasets
from nilearn import datasets, image
from nilearn.input_data import NiftiMasker
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.constants import DATA_PATH
from src.parallel import ParallelExecutor
from src.utils import clip_predict, recall_n

import pickle
import os
from nilearn import image
from tqdm import tqdm



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

preprocessing_on_samples = partial(
    preprocessing.scale, with_mean=True, with_std=True, axis=1
)

MAIN_DIR = Path("/data/parietal/store2/work/rmeudec")

embed_folder = MAIN_DIR / "retreat-neuro-llm/data/embed/"
MASK_PATH = MAIN_DIR / "large-scale-fmri-decoding/data"


def load_atlas_masked(dimension=1024):
    mask = nib.load(MASK_PATH / "task_mask.nii.gz")
    atlas = nib.load(
        datasets.fetch_atlas_difumo(
            dimension=dimension,
            resolution_mm=2,
            data_dir=str(DATA_PATH),
        ).maps
    )

    masker = NiftiMasker(mask_img=mask).fit()
    atlas_masked = masker.transform(atlas)

    return masker, atlas, atlas_masked, mask


memory = joblib.Memory(str(Path(__file__).parent / ".cache"), verbose=0)


@memory.cache
def coordinates_to_difumo(
        coordinates,
        difumo_masker,
        atlas_pinv,
        atlas_img,
        mask_img=None,
        target_affine=(4, 4, 4),
        fwhm=9.0,
        normalize=True,
):
    masker = get_masker(mask_img=mask_img, target_affine=target_affine)
    difumos, img_pmids, img_kde_resample = [], [], []

    for pmid, img in iter_coordinates_to_maps(coordinates, mask_img=masker, fwhm=fwhm):
        resampled_img = image.resample_to_img(img, atlas_img)
        masked_img = difumo_masker.transform(resampled_img)
        if normalize:
            masked_img = StandardScaler().fit_transform(masked_img.T).T

        difumo_comp = masked_img @ atlas_pinv
        difumos.append(difumo_comp)
        img_pmids.append(pmid)
        img_kde_resample.append(resampled_img)

    return pd.DataFrame(data=np.concatenate(difumos, axis=0), index=img_pmids), masker, img_kde_resample, img_pmids


def parallel_coordinates_to_difumo(
        n_jobs,
        coordinates,
        mask_img=None,
        target_affine=(4, 4, 4),
        fwhm=9.0,
        dimension=1024,
        use_cache=True,
        normalize=True,
):
    # This array split is special as we can't simply split the input dataframe
    # of coordinates, but rather split the pmids into subgroups and then reform
    # the dataframes of peaks for those pmid subgroups
    pmids = coordinates.pmid.unique()

    masker, atlas, atlas_masked, mask = load_atlas_masked(dimension=dimension)
    atlas_pinv = np.linalg.pinv(atlas_masked)

    parallel_runner = ParallelExecutor(n_jobs=n_jobs)(total=len(pmids))

    function = coordinates_to_difumo if use_cache else coordinates_to_difumo.func
    results = parallel_runner(
        delayed(function)(
            coordinates.loc[lambda df: df.pmid == pmid],
            mask_img=mask_img,
            target_affine=target_affine,
            fwhm=fwhm,
            difumo_masker=masker,
            atlas_pinv=atlas_pinv,
            atlas_img=atlas,
            normalize=normalize,
        )
        for pmid in pmids
        # for pmid in pmids[:20]
    )

    # _, sample_masker = results[0]

    maps, sample_masker, brain_maps, brain_maps_pmids = zip(*results)

    # return pd.concat([maps for maps, _ in results], axis=0), sample_masker
    return pd.concat(maps, axis=0), sample_masker, brain_maps, brain_maps_pmids



pubget_query_dir = Path(
    "/data/parietal/store2/work/rmeudec/pubget/pubget_data/query_4dfc219125dcdee26e705e1de81c3719/subset_articlesWithCoords_extractedData")


def load_coordinates(dimension=256):
    pubget_metadata = (
        pd.read_csv(pubget_query_dir / "metadata.csv")
            .loc[:, ["pmcid", "pmid"]]
            .dropna()
            .astype(int)
    )

    coordinates = (
        pd.read_csv(pubget_query_dir / "coordinates.csv")
            .merge(pubget_metadata, on="pmcid")
            .drop(columns=["pmcid"])
    )
    old_nq_coordinates = pd.read_csv(nq_datasets.fetch_peak_coordinates())
    pmids_neuroquery = old_nq_coordinates['pmid'].drop_duplicates().astype(int)

    all_coordinates = pd.concat([
        coordinates,
        old_nq_coordinates,
    ], axis=0)

    file_path = f"nq_gaussian_difumo_{dimension}.csv"
    kde_brain_maps_path = "kde_brain_maps.pkl"

    print(f"KDE directory: {Path(__file__).parent / kde_brain_maps_path}")

    # Check if both gaussian_difumo and kde_brain_maps.csv exist
    run = not ((Path(__file__).parent / file_path).exists() and (Path(__file__).parent / kde_brain_maps_path).exists())
    load_kde = False
    if run:
        gaussian_difumo, sample_masker, kde_brain_maps, kde_brain_maps_pmids = parallel_coordinates_to_difumo(
            n_jobs=-1,
            coordinates=all_coordinates,
            target_affine=(6, 6, 6),
            # target_affine=(4, 4, 4),
            dimension=dimension,
            use_cache=False,
        )
        gaussian_difumo.columns = [str(col) for col in gaussian_difumo.columns]

        # Create a dictionary to store pmids and their corresponding kde_brain_maps
        kde_brain_maps_with_pmid = {'sample_masker': sample_masker, 'kde_brain_maps_pmids': kde_brain_maps_pmids, 'kde_brain_maps': kde_brain_maps}

        # Save the Gaussian difumo results
        gaussian_difumo.to_csv(Path(__file__).parent / file_path, index=True)

        # Save the pmids and kde_brain_maps together
        with open(Path(__file__).parent / kde_brain_maps_path, 'wb') as f:
            pickle.dump(kde_brain_maps_with_pmid, f)

    else:
        # Load the Gaussian difumo results
        gaussian_difumo = pd.read_csv(Path(__file__).parent / file_path, index_col=0)

        if load_kde:
            # Load pmids and kde_brain_maps
            with open(Path(__file__).parent / kde_brain_maps_path, 'rb') as f:
                kde_brain_maps_with_pmid = pickle.load(f)
                kde_brain_maps_pmids = kde_brain_maps_with_pmid['kde_brain_maps_pmids']
                kde_brain_maps = kde_brain_maps_with_pmid['kde_brain_maps']

        else:
            print(f"You can load KDE with pmid at : {Path(__file__).parent / kde_brain_maps_path}")
            kde_brain_maps_pmids = []
            kde_brain_maps = []
            
    # Return the gaussian difumo, pmids, and the DataFrame containing the brain maps
    return gaussian_difumo, kde_brain_maps_pmids, kde_brain_maps



def load_publications_data(dimension=512, test_size=1000, train_size_control=19000, flag_train_size_control=False, key="Mistral-7B-v0.1", load_pmid_from_file = True):
    pubget_metadata = (
        pd.read_csv(pubget_query_dir / "metadata.csv")
            .loc[:, ["pmcid", "pmid"]]
            .dropna()
            .astype(int)
    )

    gpt_neo_125m_dir = Path(embed_folder / "EleutherAI/gpt-neo-125m/")
    gpt_neo_125m_finetuned_dir = Path(embed_folder / "finetuned/finetune_gptneo_long_bs_1/")
    gpt_neo_1Bm_dir = Path(embed_folder / "EleutherAI/gpt-neo-1.3B/")
    mistral_dir = Path(embed_folder / "mistralai/Mistral-7B-v0.1")
    scibert_dir = Path(embed_folder / "allenai/scibert_scivocab_uncased")
    logger.info("Load embedded data.")
    data = {
        name: {
            "title_embeddings": (
                pd.read_csv(path / "title.csv", index_col=0)
                    .set_index("pmcid")
            ),
            "abstract_embeddings": (
                pd.read_csv(path / "abstract.csv", index_col=0)
                    .set_index("pmcid")
            ),
            "body_embeddings": (
                pd.read_csv(path / "body.csv")
                    .rename(columns={"Unnamed: 0": "pmcid"})
                    .set_index("pmcid")
            ),
        }
        for name, path in [
            ("gpt-neo-125m", gpt_neo_125m_dir / "neuroquery"),
            ("gpt-neo-125m-finetuned", gpt_neo_125m_finetuned_dir / "neuroquery"),
            ("gpt-neo-1.3B", Path(embed_folder / "EleutherAI/gpt-neo-1.3B/neuroquery")),
            ("Mistral-7B-v0.1", Path(embed_folder / "mistralai/Mistral-7B-v0.1/neuroquery")),
            ("gpt-neo-1.3B-finetuned", Path(embed_folder / "finetuned/qlora_finetune/neuroquery")),
            ("scibert", Path(scibert_dir / "neuroquery")),
        ]
        if name == key
    }

    for embeddings_key in ["title_embeddings", "abstract_embeddings", "body_embeddings"]:
        data[key][embeddings_key] = (
            data[key][embeddings_key]
                .reset_index()
                .merge(pubget_metadata, how="inner", on="pmcid")
                .dropna()
                .drop(columns=["pmcid"])
                .set_index("pmid")
        )

    logger.info("Load old NQ database")
    old_nq_data = {
        name: {
            "title_embeddings": (
                pd.read_csv(path / "title.csv", index_col=0)
                    .rename(columns={"pmcid": "pmid"})  # actual pmcid stored is the pmid
                    .groupby("pmid")
                    .mean()
                    .reset_index()
                    .set_index("pmid")
            ),
            "abstract_embeddings": (
                pd.read_csv(path / "abstract.csv", index_col=0)
                    .rename(columns={"pmcid": "pmid"})  # actual pmcid stored is the pmid
                    .groupby("pmid")
                    .mean()
                    .reset_index()
                    .set_index("pmid")
            ),
            "body_embeddings": (
                pd.read_csv(path / "body.csv")
                    .rename(columns={"Unnamed: 0": "pmcid"})
                    .rename(columns={"pmcid": "pmid"})  # actual pmcid stored is the pmid
                    .groupby("pmid")
                    .mean()
                    .reset_index()
                    .set_index("pmid")
            ),
        }
        for name, path in [
            ("gpt-neo-125m", gpt_neo_125m_dir / "neuroquery_old"),
            ("gpt-neo-125m-finetuned", gpt_neo_125m_finetuned_dir / "neuroquery_old"),
            ("gpt-neo-1.3B", gpt_neo_1Bm_dir / "neuroquery_old"),
            ("Mistral-7B-v0.1", mistral_dir / "neuroquery_old"),
            ("scibert", scibert_dir / "neuroquery_old"),
        ]
        if name == key
    }

    title_embeddings = pd.concat([
        data[key]["title_embeddings"],
        old_nq_data[key]["title_embeddings"],
    ], axis=0)
    abstract_embeddings = pd.concat([
        data[key]["abstract_embeddings"],
        old_nq_data[key]["abstract_embeddings"],
    ], axis=0)
    # For the body, we aggregate the embeddings across text chunks by averaging them
    body_embeddings = pd.concat([
        (
            data[key]["body_embeddings"]
                .reset_index()
                .groupby("pmid")
                .mean()
                .reset_index()
                .set_index("pmid")
        ),
        (
            old_nq_data[key]["body_embeddings"]
                .reset_index()
                .groupby("pmid")
                .mean()
                .reset_index()
                .set_index("pmid")
        ),
    ], axis=0)

    pmids_data = set(data[key]["body_embeddings"].index.unique())
    pmids_old_data = set(old_nq_data[key]["body_embeddings"].index.unique())
    shared_pmids = pmids_data.intersection(pmids_old_data)
    filtered_old_nq_data = old_nq_data[key]["body_embeddings"].loc[~old_nq_data[key]["body_embeddings"].index.isin(shared_pmids)]
    body_embeddings_chunks = pd.concat([
    data[key]["body_embeddings"].reset_index().set_index("pmid"),
    filtered_old_nq_data.reset_index().set_index("pmid"),
    ], axis=0)
    # body_embeddings_chunks = body_embeddings_chunks.groupby("pmid")

    title_embeddings = title_embeddings[~title_embeddings.index.duplicated(keep="first")]
    abstract_embeddings = abstract_embeddings[~abstract_embeddings.index.duplicated(keep="first")]
    body_embeddings = body_embeddings[~body_embeddings.index.duplicated(keep="first")]

    selected_pmids = list(set(title_embeddings.index.unique()).intersection(abstract_embeddings.index.unique()).intersection(body_embeddings.index.unique()))
    title_embeddings = title_embeddings.loc[selected_pmids]
    abstract_embeddings = abstract_embeddings.loc[selected_pmids]
    body_embeddings = body_embeddings.loc[selected_pmids]
    # body_embeddings_chunks = body_embeddings_chunks.get_group(selected_pmids)
    body_embeddings_chunks = body_embeddings_chunks.loc[selected_pmids]

    gaussian_difumo, pmids_neuroquery, kde_brain_maps = load_coordinates(dimension)

    logger.info("Align embedded with voxels")
    selected_pmids = list(set(list(title_embeddings.index)).intersection(set(list(gaussian_difumo.index))))
    logger.info(f"Fitting on {len(selected_pmids)} samples")
    gaussian_difumo = gaussian_difumo.loc[selected_pmids]
    title_embeddings = title_embeddings.loc[selected_pmids]
    abstract_embeddings = abstract_embeddings.loc[selected_pmids]
    body_embeddings = body_embeddings.loc[selected_pmids]
    # body_embeddings_chunks = body_embeddings_chunks.get_group(selected_pmids)
    body_embeddings_chunks = body_embeddings_chunks.loc[selected_pmids]

    assert len(gaussian_difumo) > 15000, "Merge between datasets went wrong."
    assert len(gaussian_difumo) < 22000, "If len(gaussian_difumo) is around 24k, you probably have duplicates."
    assert len(title_embeddings) > 15000, "Merge between datasets went wrong."
    assert len(title_embeddings) < 22000, "If len(gaussian_difumo) is around 24k, you probably have duplicates."

    kept_indexes = np.abs(gaussian_difumo).sum(axis=1) > 0
    gaussian_difumo = gaussian_difumo[kept_indexes]
    title_embeddings = title_embeddings[kept_indexes]
    abstract_embeddings = abstract_embeddings[kept_indexes]
    body_embeddings = body_embeddings[kept_indexes]
    # body_embeddings_chunks = body_embeddings_chunks.get_group(kept_indexes)
    body_embeddings_chunks = body_embeddings_chunks[kept_indexes]

    pubget_text_data = pd.read_csv(pubget_query_dir / "text.csv")

    ### below, we want to choose the test data out of NQ, because the sota is trained on the NQ data. This is to avoide leakage.

    # Step 1: Identify PMIDs not in pmids_neuroquery
    pmids_to_include = title_embeddings.index[~title_embeddings.index.isin(pmids_old_data)]

    # Step 2: Randomly select a subset of PMIDs for the validation set based on test_size
    # np.random.seed(42)  # For reproducibility, adjust the seed as necessary
    # val_pmids = np.random.choice(pmids_to_include, size=test_size, replace=False)

    # load_pmid_from_file = False
    
    if load_pmid_from_file:
        pmid_file_path = '/data/parietal/store3/work/fghayyem/projects/MICCAI_2024/results_kde_groundtruth_left_out_articles/pmids/pmids.npy'
        val_pmids = np.load(pmid_file_path)
        print(f"Validation PMIDs loaded from: {pmid_file_path}")
    else:
        val_pmids = np.random.choice(pmids_to_include, size=test_size, replace=False)

    # Identify PMIDs for the training set as all remaining PMIDs not chosen for the validation set
    train_pmids = np.setdiff1d(title_embeddings.index, val_pmids)
    if flag_train_size_control == True:
        # Randomly choose train_size elements from train_pmids without replacement
        train_pmids = np.random.choice(train_pmids, size=train_size_control, replace=False)
        print(flag_train_size_control)
    # Step 3: Split data into training and validation sets based on PMIDs
    train_title_embeddings = title_embeddings.loc[train_pmids]
    val_title_embeddings = title_embeddings.loc[val_pmids]

    train_abstract_embeddings = abstract_embeddings.loc[train_pmids]
    val_abstract_embeddings = abstract_embeddings.loc[val_pmids]

    train_body_embeddings = body_embeddings.loc[train_pmids]
    val_body_embeddings = body_embeddings.loc[val_pmids]

    # train_body_embeddings_chunks = body_embeddings_chunks.get_group(train_pmids)
    # val_body_embeddings_chunks = body_embeddings_chunks.get_group(val_pmids)
    train_body_embeddings_chunks = body_embeddings_chunks.loc[train_pmids]
    val_body_embeddings_chunks = body_embeddings_chunks.loc[val_pmids]

    train_gaussian_difumo = gaussian_difumo.loc[train_pmids]
    val_gaussian_difumo = gaussian_difumo.loc[val_pmids]

    # def group_to_df(group):
    #     # Assuming 'group' is a DataFrameGroupBy object
    #     # Convert each group to a DataFrame and perform any necessary operations
    #     df_list = [group_data for _, group_data in group]
    #     combined_df = pd.concat(df_list)
    #     return combined_df

    # (
    #     train_pmids,
    #     val_pmids,
    #     train_gaussian_difumo,
    #     val_gaussian_difumo,
    #     train_title_embeddings,
    #     val_title_embeddings,
    #     train_abstract_embeddings,
    #     val_abstract_embeddings,
    #     train_body_embeddings,
    #     val_body_embeddings,
    # ) = train_test_split(
    #     title_embeddings.index,
    #     preprocessing_on_samples(gaussian_difumo.values),
    #     title_embeddings.values,
    #     abstract_embeddings.values,
    #     body_embeddings.values,
    #     test_size=test_size,
    #     # random_state=42,
    # )

    # print(f"type train_body_embeddings_chunks: {type(train_body_embeddings_chunks)}")
    # print(f" train_body_embeddings_chunks: {train_body_embeddings_chunks}")

    return (
        pubget_text_data,
        pubget_metadata,
        kde_brain_maps,
        train_pmids,
        val_pmids,
        train_gaussian_difumo,
        val_gaussian_difumo,
        train_title_embeddings,
        val_title_embeddings,
        train_abstract_embeddings,
        val_abstract_embeddings,
        train_body_embeddings,
        val_body_embeddings,
        # group_to_df(train_body_embeddings_chunks),
        # group_to_df(val_body_embeddings_chunks),
        # group_to_df(body_embeddings_chunks),
        train_body_embeddings_chunks,
        val_body_embeddings_chunks,
        body_embeddings_chunks,
    )


def recall_n_callback(loader, num_samples=None, device=DEVICE):
    def run_callback(model, epoch_index, n=10):
        model.eval()

        with torch.no_grad():
            text_probs = clip_predict(
                model,
                loader,
                num_samples=num_samples,
                device=device,
            )

        recall = recall_n(
            text_probs,
            np.eye(len(text_probs)),
            n_first=n,
            thresh=0.95,
            reduce_mean=True,
        )
        return recall

    return run_callback


def diagonal_callback(loader, device=DEVICE):
    # mean of the diagonal
    # diagonal dominance? ratio of diagonal over sum(non-diagonal terms)
    def run_callback(model, epoch_index):
        model.eval()

        with torch.no_grad():
            text_probs = clip_predict(
                model,
                loader,
                num_samples=None,
                device=device,
            )

        return np.mean(np.diag(text_probs))

    return run_callback

def non_diagonal_callback(loader, device=DEVICE):
    def run_callback(model, epoch_index):
        model.eval()

        with torch.no_grad():
            text_probs = clip_predict(
                model,
                loader,
                num_samples=None,
                device=device,
            )

        return np.mean(text_probs - np.diag(np.diag(text_probs)))

    return run_callback

def term_to_one_callback(loader, device=DEVICE):
    def run_callback(model, epoch_index):
        model.eval()

        with torch.no_grad():
            text_probs = clip_predict(
                model,
                loader,
                num_samples=None,
                device=device,
            )

        text_probs[text_probs > 0.9999] = 1
        text_probs[text_probs < 0.9999] = 0

        return np.mean(text_probs.sum(axis=1) / len(text_probs))

    return run_callback


if __name__ == "__main__":
    (
        train_pmids,
        test_pmids,
        train_gaussian_difumo,
        test_gaussian_difumo,
        train_title_embeddings,
        test_title_embeddings,
        train_abstract_embeddings,
        test_abstract_embeddings,
        train_keywords_embeddings,
        test_keywords_embeddings,
        train_body_embeddings,
        test_body_embeddings,
    ) = load_publications_data(
        dimension=256,
        test_size=1000,
        key="gpt-neo-125m",  # possible keys are {"gpt-neo-125m", "gpt-neo-125m-finetuned", "gpt-neo-1.3B}
    )



def extract_difumo_from_images(
    brain_images_path,
    output_csv_path,
    dimension=1024,
    target_affine=(4, 4, 4),
    fwhm=9.0,
    normalize=True,
):
    """
    Extracts DiFuMo components from brain images in the specified folder.

    Parameters:
    - brain_images_path (str or Path): Path to the folder containing .nii.gz brain images.
    - dimension (int): Dimensionality for the atlas.
    - target_affine (tuple): Target affine transformation for resampling.
    - fwhm (float): Full width at half maximum for smoothing.
    - normalize (bool): Whether to normalize the data.
    - output_csv_path (str or Path): Path to save the output CSV file with DiFuMo components.

    Returns:
    - pd.DataFrame: DataFrame containing the DiFuMo components with image filenames as indices.
    """
    # Check if the output file exists
    output_csv_path = Path(output_csv_path)
    if output_csv_path.is_file():
        print(f"File {output_csv_path} exists. Loading existing data for IBC contrast DiFuMo.")
        return pd.read_csv(output_csv_path, index_col=0)
    
    # Load the atlas and precomputed atlas pseudo-inverse
    print("Loading DiFuMo atlas ...")
    masker, atlas, atlas_masked, mask = load_atlas_masked(dimension=dimension)
    print("DiFuMo atlas loaded ...")
    atlas_pinv = np.linalg.pinv(atlas_masked)

    # Convert the path to a Path object if it's a string
    brain_images_path = Path(brain_images_path)
    
    # Initialize lists to store results
    difumos = []
    img_filenames = []

    # Process each .nii.gz file in the folder
    for img_filename in tqdm(brain_images_path.glob("*.nii.gz"), desc="Processing brain images"):
        # Load the brain image
        img = nib.load(str(img_filename))
        
        # Resample the image to match the atlas dimensions
        resampled_img = image.resample_to_img(img, atlas)
        
        # Apply the masker and get masked image data
        masked_img = masker.transform(resampled_img)
        
        # Normalize if required
        if normalize:
            masked_img = StandardScaler().fit_transform(masked_img.T).T

        # Compute DiFuMo components
        difumo_comp = masked_img @ atlas_pinv
        difumos.append(difumo_comp)
        img_filenames.append(img_filename.stem)
    # Convert results to DataFrame
    gaussian_difumo = pd.DataFrame(data=np.concatenate(difumos, axis=0), index=img_filenames)
    gaussian_difumo.index = gaussian_difumo.index.str.replace('.nii', '', regex=False)

    # Save the results to a CSV file
    gaussian_difumo.to_csv(output_csv_path, index=True)

    return gaussian_difumo

