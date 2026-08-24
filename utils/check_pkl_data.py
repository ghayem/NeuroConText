import pickle
import sys

def identify_pkl_data(filepath):
    """
    Identify the type of data stored in a pickle file.
    """
    try:
        with open(filepath, 'rb') as file:
            data = pickle.load(file)

        # Get basic type information
        data_type = type(data).__name__

        # Get additional information based on data type
        if isinstance(data, dict):
            info = f"Dictionary with {len(data)} key-value pairs"
            sample = {k: str(v)[:50] for k, v in list(data.items())[:3]}

        elif isinstance(data, list):
            info = f"List with {len(data)} items"
            sample = [str(item)[:50] for item in data[:3]]

        elif isinstance(data, tuple):
            info = f"Tuple with {len(data)} items"
            sample = [str(item)[:50] for item in data[:3]]

        elif isinstance(data, set):
            info = f"Set with {len(data)} items"
            sample = [str(item)[:50] for item in list(data)[:3]]

        elif isinstance(data, str):
            info = f"String with {len(data)} characters"
            sample = data[:100] + "..." if len(data) > 100 else data

        elif isinstance(data, (int, float, bool)):
            info = f"Primitive {data_type}: {data}"
            sample = data

        elif hasattr(data, '__dict__'):
            info = f"Object of type {data_type}"
            sample = {attr: str(value)[:50] for attr, value in data.__dict__.items()}

        else:
            info = f"Unknown type: {data_type}"
            sample = str(data)[:100]

        return data_type, info, sample

    except FileNotFoundError:
        return None, f"File '{filepath}' not found.", None
    except pickle.UnpicklingError:
        return None, "Not a valid pickle file or corrupted file.", None
    except EOFError:
        return None, "File is empty or incomplete.", None
    except Exception as e:
        return None, f"Error reading file: {str(e)}", None


def main():
    # Get filepath from command line argument
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_pkl_file>")
        sys.exit(1)

    filepath = sys.argv[1]

    print(f"\nAnalyzing: {filepath}")
    print("-" * 50)

    data_type, info, sample = identify_pkl_data(filepath)

    if data_type is None:
        print(f"❌ {info}")
        return

    print(f"✅ Data Type: {data_type}")
    print(f"📝 {info}")

    print("\nSample data (first few items):")
    print("-" * 30)

    if isinstance(sample, dict):
        for key, value in sample.items():
            print(f"  {key}: {value}")
    elif isinstance(sample, list):
        for i, item in enumerate(sample):
            print(f"  [{i}] {item}")
    else:
        print(f"  {sample}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
