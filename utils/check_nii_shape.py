import sys
from pathlib import Path
import nibabel as nib


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_nii_shape.py <path_to_file.nii.gz>")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

    # nib.load reads the header metadata without loading full image array into memory
    img = nib.load(file_path)

    print(f"File:       {file_path.name}")
    print(f"Shape:      {img.shape}")
    print(f"Data type:  {img.get_data_dtype()}")
    print(f"Voxel size: {img.header.get_zooms()}")


if __name__ == "__main__":
    main()
