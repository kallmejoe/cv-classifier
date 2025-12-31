"""
Simple PyTorch example that trains a linear classifier on TF-IDF features
and uses MPS (Apple Metal) if available on macOS (M1/M2).

Usage:
  python gpu_train_example.py --csv UpdatedResumeDataSet.csv --text-col text --label-col label

The script is intentionally minimal: it shows device detection and a tiny training loop.
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.lin = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.lin(x)


def train(args):
    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise SystemExit(f"CSV must contain columns {args.text_col} and {args.label_col}")

    texts = df[args.text_col].astype(str).fillna("").tolist()
    labels = df[args.label_col].astype(str).fillna("").tolist()

    le = LabelEncoder()
    y = le.fit_transform(labels)

    vect = TfidfVectorizer(max_features=5000)
    X = vect.fit_transform(texts)

    # convert to dense (for demo). For large corpora, use a different approach.
    X = X.toarray().astype(np.float32)
    y = np.array(y, dtype=np.int64)

    device = get_device()
    print(f"Using device: {device}")

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = LinearClassifier(X.shape[1], len(le.classes_)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu().numpy()) * xb.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {avg_loss:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": le.classes_.tolist(),
        "vectorizer_vocab": vect.vocabulary_,
    }, args.out)
    print(f"Saved model to {args.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to CSV with text and labels")
    p.add_argument("--text-col", default="text", help="Column name with text")
    p.add_argument("--label-col", default="label", help="Column name with label")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default="models/torch_mps_model.pt")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
