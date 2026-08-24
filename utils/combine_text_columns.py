import pandas as pd
import sys
from pathlib import Path

def combine_text_columns(csv_file, output_dir=None):
    """
    Combine title, abstract, and body into a single full_text column.

    Args:
        csv_file: Path to the markdown-ready CSV file
        output_dir: Output directory (default: same as input file)
    """
    try:
        # Load the CSV
        print(f"📂 Loading: {csv_file}")
        df = pd.read_csv(csv_file, encoding='utf-8')

        print(f"✅ Loaded DataFrame with shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")

        # Check if required columns exist
        required_cols = ['title', 'abstract', 'body']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"❌ Warning: Missing columns: {missing_cols}")
            print(f"   Available columns: {list(df.columns)}")
            # Continue with whatever columns exist
            available_cols = [col for col in required_cols if col in df.columns]
            if not available_cols:
                print("❌ No text columns found to combine!")
                return False
        else:
            available_cols = required_cols

        # Create full_text column
        print(f"🧹 Combining columns: {available_cols}")

        # Initialize with empty string
        df['full_text'] = ''

        # Add each column with appropriate spacing
        for col in available_cols:
            # Get column data, fill NaN with empty string
            col_data = df[col].fillna('').astype(str)

            # Add the column content with a newline separator
            # If it's the first column, just add it
            # Otherwise, add with double newline for markdown spacing
            if df['full_text'].str.len().sum() == 0:
                df['full_text'] = col_data
            else:
                df['full_text'] = df['full_text'] + '\n\n' + col_data

        # Remove any excessive whitespace
        df['full_text'] = df['full_text'].str.strip()

        # Remove rows where full_text is empty
        df = df[df['full_text'] != '']
        df = df.reset_index(drop=True)

        # Select columns to keep
        # Keep pmcid, pmid, full_text, and any other non-text columns
        columns_to_keep = []

        # Keep identifier columns if they exist
        for col in ['pmcid', 'pmid']:
            if col in df.columns:
                columns_to_keep.append(col)

        # Add full_text
        columns_to_keep.append('full_text')

        # Also keep any other columns that might be useful (but not the original text columns)
        for col in df.columns:
            if col not in ['pmcid', 'pmid', 'title', 'abstract', 'body', 'keywords', 'full_text']:
                columns_to_keep.append(col)

        # Create final DataFrame
        df_final = df[columns_to_keep]

        print(f"📊 Final shape: {df_final.shape}")
        print(f"📊 Columns: {list(df_final.columns)}")

        # Determine output path
        if output_dir is None:
            output_dir = Path(csv_file).parent
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generate output filename
        input_path = Path(csv_file)
        output_file = Path(output_dir) / f"{input_path.stem}_combined.csv"

        # Save to CSV
        print(f"💾 Saving to: {output_file}")
        df_final.to_csv(output_file, index=False, encoding='utf-8')

        # Show sample
        print("\n📋 Sample of combined text:")
        print("-" * 80)
        if len(df_final) > 0:
            sample = df_final['full_text'].iloc[0]
            print(f"Length: {len(sample)} characters")
            print(f"Preview: {sample[:500]}...")
            print(f"\nNumber of lines: {sample.count(chr(10)) + 1}")

        print("\n✅ Done!")
        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Combine title, abstract, and body into full_text column'
    )
    parser.add_argument('csv_file', help='Path to the markdown-ready CSV file')
    parser.add_argument('-o', '--output-dir', help='Output directory (optional)')

    args = parser.parse_args()

    combine_text_columns(args.csv_file, args.output_dir)


if __name__ == "__main__":
    main()
