import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import glob

def combine_train_test_files(input_dir, output_dir=None):
    """
    Combine train and test embedding files into full datasets.

    Args:
        input_dir: Directory containing train_* and test_* files
        output_dir: Output directory (default: same as input_dir)
    """
    try:
        input_path = Path(input_dir)

        if not input_path.exists():
            print(f"❌ Error: Directory '{input_dir}' not found")
            return False

        # Set output directory
        if output_dir is None:
            output_dir = input_path
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"📂 Processing files in: {input_path}")
        print(f"📁 Output directory: {output_dir}")
        print("=" * 80)

        # Find all files
        train_pkl = input_path / "train_text_embeddings.pkl"
        test_pkl = input_path / "test_text_embeddings.pkl"
        train_pmids = input_path / "train_text_embeddings_pmids.txt"
        test_pmids = input_path / "test_text_embeddings_pmids.txt"
        train_npy = input_path / "train_text_embeddings_embeddings.npy"
        test_npy = input_path / "test_text_embeddings_embeddings.npy"

        # Check if files exist
        missing_files = []
        for f in [train_pkl, test_pkl, train_pmids, test_pmids, train_npy, test_npy]:
            if not f.exists():
                missing_files.append(f.name)

        if missing_files:
            print(f"❌ Missing files: {', '.join(missing_files)}")
            return False

        print("✅ All files found!")

        # Load train data
        print("\n📊 Loading train data...")
        with open(train_pkl, 'rb') as f:
            train_df = pickle.load(f)
        train_pmids_list = []
        with open(train_pmids, 'r') as f:
            train_pmids_list = [int(line.strip()) for line in f]
        train_embeddings = np.load(train_npy)

        print(f"   Train shape: {train_embeddings.shape}")
        print(f"   Train PMIDs: {len(train_pmids_list)}")

        # Load test data
        print("\n📊 Loading test data...")
        with open(test_pkl, 'rb') as f:
            test_df = pickle.load(f)
        test_pmids_list = []
        with open(test_pmids, 'r') as f:
            test_pmids_list = [int(line.strip()) for line in f]
        test_embeddings = np.load(test_npy)

        print(f"   Test shape: {test_embeddings.shape}")
        print(f"   Test PMIDs: {len(test_pmids_list)}")

        # Combine data
        print("\n🔗 Combining train and test data...")

        # Combine embeddings
        full_embeddings = np.vstack([train_embeddings, test_embeddings])

        # Combine PMIDs
        full_pmids = train_pmids_list + test_pmids_list

        print(f"✅ Combined embeddings shape: {full_embeddings.shape}")
        print(f"✅ Combined PMIDs count: {len(full_pmids)}")

        # Create combined DataFrame (with PMID as index)
        print("\n📦 Creating combined DataFrame...")
        full_df = pd.DataFrame({
            'embedding': list(full_embeddings)
        }, index=full_pmids)

        # Create combined DataFrame (with PMID as column)
        full_df_col = pd.DataFrame({
            'pmid': full_pmids,
            'embedding': list(full_embeddings)
        })

        # Save files with full_ prefix
        print(f"\n💾 Saving files to: {output_dir}")

        # 1. Save pickle
        pkl_file = Path(output_dir) / "full_text_embeddings.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(full_df, f)
        print(f"   ✅ {pkl_file}")

        # 2. Save CSV
        csv_file = Path(output_dir) / "full_text_embeddings.csv"
        df_csv = full_df_col.copy()
        df_csv['embedding'] = df_csv['embedding'].apply(lambda x: ' '.join(map(str, x)))
        df_csv.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"   ✅ {csv_file}")

        # 3. Save numpy array
        npy_file = Path(output_dir) / "full_text_embeddings_embeddings.npy"
        np.save(npy_file, full_embeddings)
        print(f"   ✅ {npy_file}")

        # 4. Save PMIDs
        pmid_file = Path(output_dir) / "full_text_embeddings_pmids.txt"
        with open(pmid_file, 'w') as f:
            for pmid in full_pmids:
                f.write(f"{pmid}\n")
        print(f"   ✅ {pmid_file}")

        # Also save a metadata file with split info
        metadata_file = Path(output_dir) / "full_text_embeddings_metadata.txt"
        with open(metadata_file, 'w') as f:
            f.write("Full Text Embeddings Dataset\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total samples: {len(full_pmids)}\n")
            f.write(f"Embedding dimension: {full_embeddings.shape[1]}\n")
            f.write(f"Train samples: {len(train_pmids_list)}\n")
            f.write(f"Test samples: {len(test_pmids_list)}\n")
            f.write(f"PMID range: {min(full_pmids)} - {max(full_pmids)}\n")
            f.write("\nSplit indices:\n")
            f.write(f"Train: 0 to {len(train_pmids_list)-1}\n")
            f.write(f"Test: {len(train_pmids_list)} to {len(full_pmids)-1}\n")
        print(f"   ✅ {metadata_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("✅ Processing complete! Created 5 files:")
        print("=" * 80)
        print(f"1. 📦 {pkl_file}")
        print(f"2. 📊 {csv_file}")
        print(f"3. 🔢 {npy_file}")
        print(f"4. 📝 {pmid_file}")
        print(f"5. 📋 {metadata_file}")
        print("=" * 80)

        # Show sample
        print("\n📋 Sample of combined data:")
        print("-" * 80)
        for i in range(min(3, len(full_pmids))):
            split_label = "Train" if i < len(train_pmids_list) else "Test"
            print(f"  [{i}] Split: {split_label}, PMID: {full_pmids[i]}, Embedding shape: {full_embeddings[i].shape}")
            print(f"       First 5 values: {full_embeddings[i][:5]}")

        print(f"\n📊 Statistics:")
        print(f"   Total articles: {len(full_pmids)}")
        print(f"   Train/Test split: {len(train_pmids_list)} / {len(test_pmids_list)}")
        print(f"   Embedding dimension: {full_embeddings.shape[1]}")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Combine train and test embedding files into full datasets'
    )
    parser.add_argument('input_dir',
                       help='Directory containing train_* and test_* files')
    parser.add_argument('-o', '--output-dir',
                       help='Output directory (default: same as input_dir)')

    args = parser.parse_args()

    combine_train_test_files(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
