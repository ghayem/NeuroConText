#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert CSV to Pickle for NeuroConText Retrieval System.
Takes the articles_markdown_ready.csv and converts it to a pickle file
with proper columns for the retrieval system.
"""

import pandas as pd
import pickle
import argparse
from pathlib import Path
import sys
import numpy as np

def convert_csv_to_pkl(csv_path: str, output_path: str = None, sample: int = None):
    """
    Convert CSV to pickle file with proper columns.

    Args:
        csv_path: Path to the CSV file
        output_path: Path to save the pickle file (optional)
        sample: Number of rows to sample (for testing)
    """
    print(f"📂 Loading CSV: {csv_path}")

    # Check file size
    file_size = Path(csv_path).stat().st_size / (1024 * 1024)  # MB
    print(f"   File size: {file_size:.2f} MB")

    # Load CSV - handle large files with low_memory=False
    try:
        if file_size > 500:  # If >500MB, load in chunks
            print("   Large file detected, loading in chunks...")
            chunks = []
            for chunk in pd.read_csv(csv_path, chunksize=50000, low_memory=False):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"❌ Error loading CSV: {str(e)}")
        return False

    print(f"✅ Loaded {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")

    # Sample if requested
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
        print(f"   Sampled {len(df)} rows")

    # Ensure required columns exist
    required_cols = ['pmid', 'title', 'abstract', 'body']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"⚠️ Warning: Missing columns: {missing_cols}")
        # Try to find alternative column names
        for col in missing_cols:
            if col == 'pmid':
                # Try to find pmid-like column
                pmid_candidates = [c for c in df.columns if 'pmid' in c.lower() and 'pmcid' not in c.lower()]
                if pmid_candidates:
                    print(f"   Using '{pmid_candidates[0]}' as pmid")
                    df['pmid'] = df[pmid_candidates[0]]
            elif col == 'title':
                title_candidates = [c for c in df.columns if 'title' in c.lower()]
                if title_candidates:
                    print(f"   Using '{title_candidates[0]}' as title")
                    df['title'] = df[title_candidates[0]]
            elif col == 'abstract':
                abstract_candidates = [c for c in df.columns if 'abstract' in c.lower()]
                if abstract_candidates:
                    print(f"   Using '{abstract_candidates[0]}' as abstract")
                    df['abstract'] = df[abstract_candidates[0]]
            elif col == 'body':
                body_candidates = [c for c in df.columns if 'body' in c.lower() or 'full_text' in c.lower()]
                if body_candidates:
                    print(f"   Using '{body_candidates[0]}' as body")
                    df['body'] = df[body_candidates[0]]

    # Check if we have all required columns now
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Still missing required columns: {missing_cols}")
        print("   Available columns:", list(df.columns))
        return False

    # Clean and convert pmid to int, handling NaN values
    print("\n🔧 Cleaning data...")

    # First, drop rows where pmid is NaN
    before_drop = len(df)
    df = df[df['pmid'].notna()]
    after_drop = len(df)
    if before_drop != after_drop:
        print(f"   Dropped {before_drop - after_drop} rows with NaN pmid")

    # Convert pmid to int, handling potential float issues
    try:
        # First convert to float, then to int (handles NaN)
        df['pmid'] = df['pmid'].astype(float).astype(int)
        print(f"   Converted pmid to int")
    except (ValueError, TypeError) as e:
        print(f"   Warning: Could not convert directly: {e}")
        # Try extracting numbers from strings
        try:
            df['pmid'] = df['pmid'].astype(str).str.extract(r'(\d+)')[0]
            df['pmid'] = pd.to_numeric(df['pmid'], errors='coerce')
            df = df[df['pmid'].notna()]
            df['pmid'] = df['pmid'].astype(int)
            print(f"   Extracted numeric pmid from strings")
        except Exception as e2:
            print(f"❌ Failed to convert pmid: {e2}")
            return False

    # Clean text columns
    text_cols = ['title', 'abstract', 'body']
    if 'keywords' in df.columns:
        text_cols.append('keywords')

    for col in text_cols:
        if col in df.columns:
            # Fill NaN with empty string
            df[col] = df[col].fillna('')
            # Convert to string
            df[col] = df[col].astype(str)
            # Clean whitespace
            df[col] = df[col].str.strip()

    # Remove rows where title is empty
    before_title_drop = len(df)
    df = df[df['title'] != '']
    after_title_drop = len(df)
    if before_title_drop != after_title_drop:
        print(f"   Dropped {before_title_drop - after_title_drop} rows with empty title")

    # Remove any remaining rows with NaN in key columns
    df = df.dropna(subset=['pmid', 'title', 'abstract'])

    print(f"   Final row count: {len(df)}")

    # Convert output_path to Path if string
    if output_path is None:
        input_path = Path(csv_path)
        output_path = input_path.parent / f"{input_path.stem}.pkl"
    else:
        output_path = Path(output_path)

    # Save as pickle
    print(f"\n💾 Saving to: {output_path}")

    try:
        # Use protocol 4 for better compatibility and compression
        with open(output_path, 'wb') as f:
            pickle.dump(df, f, protocol=4)

        # Also save a compressed version
        compressed_path = output_path.with_suffix('.pkl.gz')
        print(f"💾 Saving compressed version: {compressed_path}")
        with open(compressed_path, 'wb') as f:
            pickle.dump(df, f, protocol=4)

    except Exception as e:
        print(f"❌ Error saving pickle: {str(e)}")
        return False

    # Print statistics
    print("\n" + "=" * 80)
    print("✅ Conversion complete!")
    print("=" * 80)
    print(f"📊 Total articles: {len(df)}")
    print(f"📊 Columns: {list(df.columns)}")
    print(f"📊 PMID range: {df['pmid'].min()} - {df['pmid'].max()}")
    print(f"📊 PMID dtype: {df['pmid'].dtype}")

    # Show sample
    print("\n📋 Sample of first row:")
    print("-" * 80)
    row = df.iloc[0]
    for col in df.columns[:6]:  # Show first 6 columns
        if col in df.columns:
            value = str(row[col])[:100]
            if len(str(row[col])) > 100:
                value += "..."
            print(f"   {col}: {value}")

    # Check for pmcid column
    if 'pmcid' in df.columns:
        non_empty_pmcid = (df['pmcid'] != '').sum()
        print(f"\n📌 'pmcid' column found - {non_empty_pmcid}/{len(df)} rows have values")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert CSV to Pickle for NeuroConText Retrieval System'
    )
    parser.add_argument(
        'csv_file',
        help='Path to the CSV file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output pickle file path (optional)'
    )
    parser.add_argument(
        '-s', '--sample',
        type=int,
        help='Number of rows to sample (for testing)'
    )

    args = parser.parse_args()

    success = convert_csv_to_pkl(args.csv_file, args.output, args.sample)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
