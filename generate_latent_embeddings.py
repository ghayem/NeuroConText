import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from torch.utils.data import DataLoader, TensorDataset

# Import the model architecture (assumes layers.py and training.py are available)
from layers import ClipModel, ProjectionHead, ResidualHead
from losses import ClipLoss

def load_model(checkpoint_path, output_size, device):
    """
    Load the trained model from checkpoint.
    """
    print(f"📂 Loading model from: {checkpoint_path}")

    # Recreate the model architecture (must match training)
    model = ClipModel(
        image_model=torch.nn.Sequential(
            ResidualHead(output_size, dropout=0.6),
            ResidualHead(output_size, dropout=0.6),
            ResidualHead(output_size, dropout=0.6),
        ),
        text_model=torch.nn.Sequential(
            ProjectionHead(4096, output_size, dropout=0.6),  # 4096 is the text embedding dimension
            ResidualHead(output_size, dropout=0.6),
            ResidualHead(output_size, dropout=0.6),
        ),
        logit_scale=10,
        logit_bias=None,
    )

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully!")
    return model

def process_embeddings(model, text_embeddings_pkl, output_pkl, batch_size=128, device="cuda"):
    """
    Process text embeddings through the model and save latent embeddings.
    """
    try:
        print(f"📂 Loading text embeddings from: {text_embeddings_pkl}")

        # Load the text embeddings DataFrame
        with open(text_embeddings_pkl, 'rb') as f:
            df = pickle.load(f)

        print(f"✅ Loaded DataFrame with shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        print(f"📊 Index (PMIDs): {len(df.index)}")

        # Extract embeddings and PMIDs
        if isinstance(df, pd.DataFrame):
            # Check if embeddings are in a column or if the DataFrame is structured differently
            if 'embedding' in df.columns:
                # Format from our combine script
                text_embeddings = np.array(df['embedding'].tolist())
                pmids = df.index.tolist()
            elif len(df.shape) == 2 and df.shape[1] == 4096:
                # If it's a DataFrame where each row is an embedding
                text_embeddings = df.values
                pmids = df.index.tolist()
            else:
                # Try to infer structure
                print(f"⚠️ Unknown DataFrame structure. Attempting to extract...")
                # If the first column is embedding-like
                first_col = df.iloc[:, 0]
                if isinstance(first_col.iloc[0], np.ndarray):
                    text_embeddings = np.array([x for x in df.iloc[:, 0]])
                    pmids = df.index.tolist()
                else:
                    # Assume all columns are embeddings
                    text_embeddings = df.values
                    pmids = df.index.tolist()
        else:
            # If it's just a numpy array or list
            text_embeddings = np.array(df)
            pmids = list(range(len(text_embeddings)))

        print(f"📊 Text embeddings shape: {text_embeddings.shape}")
        print(f"📊 Number of PMIDs: {len(pmids)}")

        # Create dataloader
        print(f"🔄 Creating dataloader with batch size {batch_size}...")
        dataset = TensorDataset(torch.from_numpy(text_embeddings).float())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Process through model
        print(f"🧠 Processing through model on {device}...")
        latent_embeddings = []

        with torch.no_grad():
            for batch_idx, (text_emb,) in enumerate(loader):
                text_emb = text_emb.to(device)

                # Get text embeddings from the model
                # The model's text_model expects input and outputs latent embeddings
                latent = model.text_model(text_emb)
                latent_embeddings.append(latent.cpu().numpy())

                if (batch_idx + 1) % 100 == 0:
                    print(f"   Processed {batch_idx + 1} batches...")

        # Concatenate all latent embeddings
        latent_embeddings = np.vstack(latent_embeddings)
        print(f"✅ Latent embeddings shape: {latent_embeddings.shape}")

        # Create DataFrame with PMIDs as index
        result_df = pd.DataFrame({
            'embedding': list(latent_embeddings)
        }, index=pmids)

        # Save to pickle
        print(f"💾 Saving latent embeddings to: {output_pkl}")
        with open(output_pkl, 'wb') as f:
            pickle.dump(result_df, f)

        # Also save as numpy array and PMIDs for convenience
        output_dir = Path(output_pkl).parent
        output_name = Path(output_pkl).stem

        # Save numpy array
        npy_file = output_dir / f"{output_name}_latent_embeddings.npy"
        np.save(npy_file, latent_embeddings)
        print(f"💾 Saved numpy array: {npy_file}")

        # Save PMIDs
        pmid_file = output_dir / f"{output_name}_pmids.txt"
        with open(pmid_file, 'w') as f:
            for pmid in pmids:
                f.write(f"{pmid}\n")
        print(f"💾 Saved PMIDs: {pmid_file}")

        print("\n" + "=" * 80)
        print("✅ Processing complete!")
        print("=" * 80)
        print(f"📦 Output pickle: {output_pkl}")
        print(f"   - Index: PMIDs ({len(pmids)} samples)")
        print(f"   - Column: 'embedding' (shape: {latent_embeddings.shape[1]})")
        print(f"📊 Statistics:")
        print(f"   - Input dimension: {text_embeddings.shape[1]}")
        print(f"   - Output dimension: {latent_embeddings.shape[1]}")
        print(f"   - Total samples processed: {len(pmids)}")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Process text embeddings through trained model'
    )
    parser.add_argument('--checkpoint', '-c',
                       required=True,
                       help='Path to the best_val.pt checkpoint file')
    parser.add_argument('--input-pkl', '-i',
                       required=True,
                       help='Path to the input pickle file containing text embeddings')
    parser.add_argument('--output-pkl', '-o',
                       required=True,
                       help='Path to the output pickle file for latent embeddings')
    parser.add_argument('--output-size', '-s',
                       type=int,
                       default=512,
                       help='Output size of the latent space (default: 512)')
    parser.add_argument('--batch-size', '-b',
                       type=int,
                       default=128,
                       help='Batch size for processing (default: 128)')
    parser.add_argument('--device', '-d',
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (default: cuda if available else cpu)')

    args = parser.parse_args()

    # Load model
    model = load_model(args.checkpoint, args.output_size, args.device)

    # Process embeddings
    success = process_embeddings(
        model,
        args.input_pkl,
        args.output_pkl,
        batch_size=args.batch_size,
        device=args.device
    )

    if success:
        print("\n🎉 Done!")
        sys.exit(0)
    else:
        print("\n❌ Failed to process embeddings!")
        sys.exit(1)

if __name__ == "__main__":
    main()
