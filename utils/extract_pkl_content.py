import pickle
import sys
import pandas as pd
from pathlib import Path

def output_pkl_content(filepath, output_file=None, n_rows=5):
    """
    Read a pickle file and output its content (top rows) to a text file.

    Args:
        filepath: Path to the .pkl file
        output_file: Path to output text file (if None, auto-generates)
        n_rows: Number of rows to display (default: 5)
    """
    try:
        # Load the pickle file
        with open(filepath, 'rb') as file:
            data = pickle.load(file)

        # Generate output filename if not provided
        if output_file is None:
            input_path = Path(filepath)
            output_file = input_path.stem + "_content.txt"

        # Handle different data types
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Content of: {filepath}\n")
            f.write(f"Data Type: {type(data).__name__}\n")
            f.write("=" * 80 + "\n\n")

            # Check if it's a pandas DataFrame
            if isinstance(data, pd.DataFrame):
                f.write(f"DataFrame shape: {data.shape}\n")
                f.write(f"Columns: {list(data.columns)}\n")
                f.write(f"Index: {data.index}\n\n")
                f.write(f"First {n_rows} rows:\n")
                f.write("-" * 80 + "\n")
                f.write(data.head(n_rows).to_string())
                f.write("\n\n")
                f.write("Column data types:\n")
                f.write(data.dtypes.to_string())

            # Check if it's a pandas Series
            elif isinstance(data, pd.Series):
                f.write(f"Series length: {len(data)}\n")
                f.write(f"Index: {data.index}\n\n")
                f.write(f"First {n_rows} rows:\n")
                f.write("-" * 80 + "\n")
                f.write(data.head(n_rows).to_string())

            # Handle dictionaries
            elif isinstance(data, dict):
                f.write(f"Dictionary with {len(data)} key-value pairs\n\n")
                f.write("Keys and sample values:\n")
                f.write("-" * 80 + "\n")
                for i, (key, value) in enumerate(list(data.items())[:n_rows]):
                    f.write(f"{i+1}. {key}: {str(value)[:200]}...\n" if len(str(value)) > 200 else f"{i+1}. {key}: {value}\n")

            # Handle lists, tuples, sets
            elif isinstance(data, (list, tuple, set)):
                data_list = list(data)
                f.write(f"{type(data).__name__} with {len(data_list)} items\n\n")
                f.write(f"First {min(n_rows, len(data_list))} items:\n")
                f.write("-" * 80 + "\n")
                for i, item in enumerate(data_list[:n_rows]):
                    f.write(f"[{i}] {str(item)[:200]}\n")
                    if len(str(item)) > 200:
                        f.write("    ... (truncated)\n")

            # Handle numpy arrays
            elif hasattr(data, 'shape') and hasattr(data, 'dtype'):
                import numpy as np
                f.write(f"NumPy array shape: {data.shape}\n")
                f.write(f"Data type: {data.dtype}\n\n")
                f.write(f"First {n_rows} rows:\n")
                f.write("-" * 80 + "\n")
                np.set_printoptions(threshold=200, linewidth=200)
                f.write(str(data[:min(n_rows, len(data))]))

            # Handle custom objects
            elif hasattr(data, '__dict__'):
                f.write(f"Object of type {type(data).__name__}\n")
                f.write(f"Attributes:\n")
                f.write("-" * 80 + "\n")
                for attr, value in data.__dict__.items():
                    f.write(f"  {attr}: {str(value)[:200]}\n")
                    if len(str(value)) > 200:
                        f.write("    ... (truncated)\n")

            # Handle primitive types
            else:
                f.write(f"Value:\n")
                f.write("-" * 80 + "\n")
                f.write(str(data)[:1000])

            f.write("\n\n" + "=" * 80 + "\n")
            f.write(f"Total data saved to: {output_file}\n")

        print(f"✅ Content saved to: {output_file}")
        print(f"📊 Data Type: {type(data).__name__}")

    except FileNotFoundError:
        print(f"❌ File '{filepath}' not found.")
    except pickle.UnpicklingError:
        print(f"❌ Not a valid pickle file or corrupted file.")
    except EOFError:
        print(f"❌ File is empty or incomplete.")
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract content from pickle file to text')
    parser.add_argument('filepath', help='Path to the .pkl file')
    parser.add_argument('-o', '--output', help='Output text file path (optional)')
    parser.add_argument('-n', '--rows', type=int, default=5, help='Number of rows/items to show (default: 5)')

    args = parser.parse_args()

    output_pkl_content(args.filepath, args.output, args.rows)


if __name__ == "__main__":
    main()
