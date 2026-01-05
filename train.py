import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.trainer import train_model


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TRAINING RESUME CLASSIFIER")
    print("=" * 70 + "\n")

    # Train with sensible defaults
    results, model, extractor = train_model(dataset_mode="clean")

    print("\n" + "=" * 70)
    print(f"✓ Training Complete! Accuracy: {results['accuracy']*100:.2f}%")
    print("=" * 70 + "\n")
