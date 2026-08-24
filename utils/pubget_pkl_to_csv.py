import pickle
import pandas as pd
import sys
import re
from pathlib import Path
import os

def clean_text_for_csv(text):
    """
    Clean text for regular CSV - removes excessive whitespace but keeps structure.
    """
    if not isinstance(text, str):
        return text

    # Remove excessive whitespace
    text = text.strip()
    text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce 3+ newlines to 2
    text = re.sub(r' {2,}', ' ', text)      # Remove multiple spaces

    # Remove empty lines
    lines = text.split('\n')
    lines = [line for line in lines if line.strip()]
    text = '\n'.join(lines)

    return text

def clean_text_for_markdown(text):
    """
    Clean text while preserving markdown formatting.
    """
    if not isinstance(text, str):
        return text

    # Remove excessive whitespace but preserve markdown structure
    text = text.strip()
    text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce 3+ newlines to 2

    # Keep markdown headers even if they're empty lines
    lines = text.split('\n')
    lines = [line for line in lines if line.strip() or line.strip().startswith('#')]
    text = '\n'.join(lines)

    return text

def escape_newlines_for_markdown(text):
    """
    Escape newlines for markdown table compatibility.
    """
    if not isinstance(text, str):
        return text
    return text.replace('\n', '\\n')

def process_pkl_to_both_csvs(filepath, output_dir=None, output_prefix=None):
    """
    Process pickle file and save as both regular CSV and markdown-friendly CSV.

    Args:
        filepath: Path to the .pkl file
        output_dir: Directory to save output files (default: current directory)
        output_prefix: Prefix for output filenames (default: stem of input file)
    """
    try:
        print(f"📂 Loading: {filepath}")
        with open(filepath, 'rb') as file:
            data = pickle.load(file)

        if not isinstance(data, pd.DataFrame):
            print(f"❌ Error: File contains {type(data).__name__}, not a DataFrame")
            return False

        print(f"✅ Loaded DataFrame with shape: {data.shape}")
        print(f"📊 Columns: {list(data.columns)}")

        # Create copies for each output
        df_csv = data.copy()
        df_markdown = data.copy()

        # Clean text columns
        text_columns = df_csv.select_dtypes(include=['object']).columns
        print(f"🧹 Cleaning {len(text_columns)} text columns...")

        for col in text_columns:
            print(f"   Processing: {col}")
            # Regular CSV version
            df_csv[col] = df_csv[col].apply(clean_text_for_csv)
            # Markdown version
            df_markdown[col] = df_markdown[col].apply(clean_text_for_markdown)

        # Remove rows with empty titles (optional)
        if 'title' in df_csv.columns:
            df_csv = df_csv[df_csv['title'].astype(str).str.strip() != '']
            df_markdown = df_markdown[df_markdown['title'].astype(str).str.strip() != '']
            print(f"🗑️ Removed rows with empty titles")

        df_csv = df_csv.reset_index(drop=True)
        df_markdown = df_markdown.reset_index(drop=True)

        # Set output directory
        if output_dir is None:
            output_dir = os.getcwd()
        else:
            # Create directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generate output filenames
        if output_prefix is None:
            input_path = Path(filepath)
            output_prefix = input_path.stem

        # Create full paths
        csv_file = Path(output_dir) / f"{output_prefix}_cleaned.csv"
        markdown_file = Path(output_dir) / f"{output_prefix}_markdown_ready.csv"
        markdown_escaped_file = Path(output_dir) / f"{output_prefix}_markdown_escaped.csv"

        # Save regular CSV (with newlines preserved)
        print(f"💾 Saving regular CSV: {csv_file}")
        df_csv.to_csv(csv_file, index=False, encoding='utf-8')

        # Save markdown CSV (with newlines preserved for markdown rendering)
        print(f"💾 Saving markdown CSV: {markdown_file}")
        df_markdown.to_csv(markdown_file, index=False, encoding='utf-8')

        # Save markdown-escaped CSV (with \n escaped for tables)
        print(f"💾 Saving markdown-escaped CSV: {markdown_escaped_file}")
        df_escaped = df_markdown.copy()
        for col in text_columns:
            df_escaped[col] = df_escaped[col].apply(escape_newlines_for_markdown)
        df_escaped.to_csv(markdown_escaped_file, index=False, encoding='utf-8')

        print("\n" + "=" * 80)
        print("✅ Processing complete! Created 3 files:")
        print("=" * 80)
        print(f"1. 📄 {csv_file}")
        print(f"   → Regular CSV with clean text (newlines preserved)")
        print(f"2. 📝 {markdown_file}")
        print(f"   → Markdown-friendly CSV (newlines preserved for markdown rendering)")
        print(f"3. 🔧 {markdown_escaped_file}")
        print(f"   → Markdown-escaped CSV (newlines as \\n for tables)")
        print("=" * 80)
        print(f"📊 Final shape: {df_csv.shape}")

        # Show sample
        print("\n📋 Sample comparison:")
        print("-" * 80)
        if 'abstract' in df_csv.columns and len(df_csv) > 0:
            print("Regular CSV (first 200 chars):")
            print(f"  {df_csv['abstract'].iloc[0][:200]}...")
            print("\nMarkdown CSV (first 200 chars):")
            print(f"  {df_markdown['abstract'].iloc[0][:200]}...")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert pickle DataFrame to both regular and markdown CSVs'
    )
    parser.add_argument('filepath', help='Path to the .pkl file')
    parser.add_argument('-o', '--output-dir',
                       help='Output directory (default: current directory)')
    parser.add_argument('-p', '--output-prefix',
                       help='Output file prefix (default: stem of input file)')

    args = parser.parse_args()

    process_pkl_to_both_csvs(args.filepath, args.output_dir, args.output_prefix)

if __name__ == "__main__":
    main()
