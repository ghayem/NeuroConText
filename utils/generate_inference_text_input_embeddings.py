import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys

def combine_embeddings_with_pmids(embeddings_file, pmid_index_file, output_dir=None, output_name=None):
    """
    Combine embeddings with PMIDs from index file.

    Args:
        embeddings_file: Path to the pickle file containing embeddings (ndarray)
        pmid_index_file: Path to the pickle file containing DataFrame with PMIDs as index
        output_dir: Directory to save output files (default: current directory)
        output_name: Name prefix for output files (default: 'embeddings_with_pmids')
    """
    try:
        print("📂 Loading files...")

        # Load embeddings
        print(f"   Loading embeddings from: {embeddings_file}")
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)

        # Load PMID index DataFrame
        print(f"   Loading PMID index from: {pmid_index_file}")
        with open(pmid_index_file, 'rb') as f:
            pmid_df = pickle.load(f)

        # Validate data types
        if not isinstance(embeddings, np.ndarray):
            print(f"❌ Error: Embeddings file contains {type(embeddings).__name__}, not ndarray")
            return False

        if not isinstance(pmid_df, pd.DataFrame):
            print(f"❌ Error: PMID index file contains {type(pmid_df).__name__}, not DataFrame")
            return False

        print(f"✅ Embeddings shape: {embeddings.shape}")
        print(f"✅ PMID DataFrame shape: {pmid_df.shape}")
        print(f"✅ PMID index length: {len(pmid_df.index)}")

        # Check if shapes match
        if embeddings.shape[0] != len(pmid_df.index):
            print(f"⚠️ Warning: Embeddings rows ({embeddings.shape[0]}) don't match PMID index length ({len(pmid_df.index)})")
            print(f"   Using the smaller size: {min(embeddings.shape[0], len(pmid_df.index))}")
            min_len = min(embeddings.shape[0], len(pmid_df.index))
            embeddings = embeddings[:min_len]
            pmid_df = pmid_df.iloc[:min_len]

        # Create mapping from index to PMID
        print("\n🔗 Creating mapping...")
        pmids = pmid_df.index.tolist()

        # Create final structure
        print("📊 Building final DataFrame...")

        # Option 1: Save as DataFrame with PMID as index and embeddings as column
        df_embeddings = pd.DataFrame({
            'embedding': list(embeddings)
        }, index=pmids)

        # Option 2: Save as DataFrame with PMID as a column (for easier CSV export)
        df_embeddings_col = pd.DataFrame({
            'pmid': pmids,
            'embedding': list(embeddings)
        })

        # Set output directory
        if output_dir is None:
            output_dir = Path.cwd()
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Set output name
        if output_name is None:
            output_name = "embeddings_with_pmids"

        # Save as pickle (preserves embedding vectors)
        pickle_file = Path(output_dir) / f"{output_name}.pkl"
        print(f"💾 Saving pickle to: {pickle_file}")
        with open(pickle_file, 'wb') as f:
            pickle.dump(df_embeddings, f)

        # Save as CSV (embeddings will be stored as strings)
        csv_file = Path(output_dir) / f"{output_name}.csv"
        print(f"💾 Saving CSV to: {csv_file}")

        # For CSV, convert embeddings to strings
        df_csv = df_embeddings_col.copy()
        df_csv['embedding'] = df_csv['embedding'].apply(lambda x: ' '.join(map(str, x)))
        df_csv.to_csv(csv_file, index=False, encoding='utf-8')

        # Save as NPY (numpy array with corresponding PMIDs as separate file)
        npy_file = Path(output_dir) / f"{output_name}_embeddings.npy"
        print(f"💾 Saving numpy array to: {npy_file}")
        np.save(npy_file, embeddings)

        # Save PMIDs separately as well
        pmid_file = Path(output_dir) / f"{output_name}_pmids.txt"
        print(f"💾 Saving PMIDs to: {pmid_file}")
        with open(pmid_file, 'w') as f:
            for pmid in pmids:
                f.write(f"{pmid}\n")

        print("\n" + "=" * 80)
        print("✅ Processing complete! Created 4 files:")
        print("=" * 80)
        print(f"1. 📦 {pickle_file}")
        print(f"   → DataFrame with PMID as index and embedding column")
        print(f"2. 📊 {csv_file}")
        print(f"   → CSV with pmid and embedding (as string)")
        print(f"3. 🔢 {npy_file}")
        print(f"   → Raw numpy array of embeddings")
        print(f"4. 📝 {pmid_file}")
        print(f"   → List of PMIDs in order")
        print("=" * 80)

        # Show sample
        print("\n📋 Sample of first 5 rows:")
        print("-" * 80)
        sample_pmids = pmids[:5]
        sample_embeddings = embeddings[:5]
        for i, (pmid, emb) in enumerate(zip(sample_pmids, sample_embeddings)):
            print(f"  [{i}] PMID: {pmid}, Embedding shape: {emb.shape}, First 5 values: {emb[:5]}")

        # Stats
        print(f"\n📊 Statistics:")
        print(f"   Total articles: {len(pmids)}")
        print(f"   Embedding dimension: {embeddings.shape[1] if len(embeddings.shape) > 1 else 1}")
        print(f"   PMID range: {min(pmids)} - {max(pmids)}")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Combine embeddings with PMIDs from index file'
    )
    parser.add_argument('embeddings_file',
                       help='Path to pickle file containing embeddings (ndarray)')
    parser.add_argument('pmid_index_file',
                       help='Path to pickle file containing DataFrame with PMIDs as index')
    parser.add_argument('-o', '--output-dir',
                       help='Output directory (default: current directory)')
    parser.add_argument('-n', '--output-name',
                       help='Output file name prefix (default: embeddings_with_pmids)')

    args = parser.parse_args()

    combine_embeddings_with_pmids(
        args.embeddings_file,
        args.pmid_index_file,
        args.output_dir,
        args.output_name
    )


if __name__ == "__main__":
    main()
