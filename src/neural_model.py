"""Neural network model for GPU-accelerated resume classification.

This module provides a PyTorch-based neural network classifier that can
utilize NVIDIA CUDA GPUs for accelerated training and inference.

The neural model works alongside the existing sklearn-based classifiers,
providing an optional GPU-accelerated alternative when:
- Large datasets need faster training
- GPU hardware is available
- Better accuracy is desired through deep learning

Usage:
    from src.neural_model import NeuralClassifier, train_neural_model

    # Train a neural classifier
    model, history = train_neural_model(X_train, y_train, X_val, y_val, num_classes)

    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
"""

import os
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import scipy.sparse

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore

from src.gpu_utils import get_device, optimize_cuda_settings, clear_gpu_memory, is_torch_available


class ResumeClassifierNet(nn.Module):
    """
    Neural network for resume classification.

    Architecture:
    - Input layer matching TF-IDF feature dimension
    - Multiple hidden layers with batch normalization and dropout
    - Output layer with softmax for multi-class classification

    The architecture is designed for sparse TF-IDF input features and
    can be trained efficiently on GPU.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = None,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Initialize the neural network.

        Args:
            input_dim: Number of input features (vocabulary size)
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        if hidden_dims is None:
            # Default architecture scales with input size
            hidden_dims = [512, 256, 128]

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(nn.ReLU(inplace=True))

            # Dropout
            layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions with softmax."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


class NeuralClassifier:
    """
    Sklearn-compatible wrapper for PyTorch neural network classifier.

    This class provides an sklearn-like interface (fit, predict, predict_proba)
    that can be used interchangeably with other sklearn classifiers.

    Attributes:
        model: The underlying PyTorch model
        device: The compute device (cuda, mps, or cpu)
        classes_: Array of class labels (set after fitting)
        is_fitted_: Whether the model has been trained
    """

    def __init__(
        self,
        hidden_dims: List[int] = None,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 50,
        early_stopping_patience: int = 5,
        device: Optional[str] = None,
        verbose: bool = True,
        random_state: int = 42
    ):
        """
        Initialize the neural classifier.

        Args:
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            learning_rate: Initial learning rate
            batch_size: Training batch size
            epochs: Maximum number of training epochs
            early_stopping_patience: Epochs to wait before early stopping
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
            verbose: Whether to print training progress
            random_state: Random seed for reproducibility
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for NeuralClassifier. "
                "Install with: pip install torch"
            )

        self.hidden_dims = hidden_dims or [512, 256, 128]
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        self.random_state = random_state

        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

        # Set device
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)

        # Apply CUDA optimizations
        if self.device.type == 'cuda':
            optimize_cuda_settings()

        self.model: Optional[ResumeClassifierNet] = None
        self.classes_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False
        self.training_history_: Dict[str, List[float]] = {}

    def _to_tensor(
        self,
        X: Union[np.ndarray, scipy.sparse.spmatrix]
    ) -> torch.Tensor:
        """Convert input to PyTorch tensor."""
        if scipy.sparse.issparse(X):
            X = X.toarray()

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        return torch.tensor(X, dtype=torch.float32)

    def _create_data_loader(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        shuffle: bool = False
    ) -> DataLoader:
        """Create a DataLoader from tensors."""
        if y is not None:
            dataset = TensorDataset(X, y)
        else:
            dataset = TensorDataset(X)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Keep data loading on main thread for compatibility
            pin_memory=(self.device.type == 'cuda')
        )

    def fit(
        self,
        X: Union[np.ndarray, scipy.sparse.spmatrix],
        y: np.ndarray,
        X_val: Optional[Union[np.ndarray, scipy.sparse.spmatrix]] = None,
        y_val: Optional[np.ndarray] = None,
        class_weight: Optional[Dict[int, float]] = None
    ) -> 'NeuralClassifier':
        """
        Train the neural network.

        Args:
            X: Training features (sparse or dense)
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            class_weight: Optional class weights for imbalanced data

        Returns:
            self: Fitted classifier
        """
        # Store unique classes
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)

        # Create label mapping
        label_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_encoded = np.array([label_to_idx[label] for label in y])

        # Convert to tensors
        X_tensor = self._to_tensor(X)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)

        input_dim = X_tensor.shape[1]

        # Create model
        self.model = ResumeClassifierNet(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm
        ).to(self.device)

        # Create data loader
        train_loader = self._create_data_loader(X_tensor, y_tensor, shuffle=True)

        # Validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            y_val_encoded = np.array([label_to_idx[label] for label in y_val])
            X_val_tensor = self._to_tensor(X_val)
            y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)
            val_loader = self._create_data_loader(X_val_tensor, y_val_tensor, shuffle=False)

        # Loss function with optional class weights
        if class_weight is not None:
            weights = torch.tensor(
                [class_weight.get(i, 1.0) for i in range(num_classes)],
                dtype=torch.float32
            ).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )

        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=self.verbose
        )

        # Training history
        self.training_history_ = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        # Early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        if self.verbose:
            print(f"\nTraining on {self.device}...")
            print(f"Input dim: {input_dim}, Classes: {num_classes}")
            print(f"Architecture: {self.hidden_dims}")
            print("-" * 60)

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()

            avg_train_loss = train_loss / train_total
            train_acc = train_correct / train_total

            self.training_history_['train_loss'].append(avg_train_loss)
            self.training_history_['train_acc'].append(train_acc)

            # Validation phase
            val_loss = 0.0
            val_acc = 0.0

            if val_loader is not None:
                self.model.eval()
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)

                        val_loss += loss.item() * batch_X.size(0)
                        _, predicted = outputs.max(1)
                        val_total += batch_y.size(0)
                        val_correct += predicted.eq(batch_y).sum().item()

                val_loss = val_loss / val_total
                val_acc = val_correct / val_total

                self.training_history_['val_loss'].append(val_loss)
                self.training_history_['val_acc'].append(val_acc)

                # Learning rate scheduling
                scheduler.step(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if self.verbose:
                    print(f"Epoch {epoch+1}/{self.epochs} - "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    break
            else:
                if self.verbose:
                    print(f"Epoch {epoch+1}/{self.epochs} - "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Restore best model if using validation
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)

        self.is_fitted_ = True

        if self.verbose:
            print("-" * 60)
            print(f"Training complete. Best val loss: {best_val_loss:.4f}")

        return self

    def predict(
        self,
        X: Union[np.ndarray, scipy.sparse.spmatrix]
    ) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features to predict

        Returns:
            Array of predicted class labels
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        self.model.eval()
        X_tensor = self._to_tensor(X).to(self.device)

        predictions = []

        # Predict in batches
        with torch.no_grad():
            for i in range(0, X_tensor.shape[0], self.batch_size):
                batch = X_tensor[i:i + self.batch_size]
                outputs = self.model(batch)
                _, predicted = outputs.max(1)
                predictions.extend(predicted.cpu().numpy())

        # Convert indices back to original labels
        return np.array([self.classes_[idx] for idx in predictions])

    def predict_proba(
        self,
        X: Union[np.ndarray, scipy.sparse.spmatrix]
    ) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features to predict

        Returns:
            Array of class probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        self.model.eval()
        X_tensor = self._to_tensor(X).to(self.device)

        probabilities = []

        with torch.no_grad():
            for i in range(0, X_tensor.shape[0], self.batch_size):
                batch = X_tensor[i:i + self.batch_size]
                probs = self.model.predict_proba(batch)
                probabilities.append(probs.cpu().numpy())

        return np.vstack(probabilities)

    def score(
        self,
        X: Union[np.ndarray, scipy.sparse.spmatrix],
        y: np.ndarray
    ) -> float:
        """
        Return accuracy score.

        Args:
            X: Features
            y: True labels

        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def save(self, path: str) -> None:
        """Save model to disk."""
        if not self.is_fitted_:
            raise ValueError("Cannot save unfitted model")

        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'classes': self.classes_,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'input_dim': self.model.input_dim,
            'num_classes': self.model.num_classes,
            'training_history': self.training_history_
        }

        torch.save(save_dict, path)

    def load(self, path: str) -> 'NeuralClassifier':
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)

        self.classes_ = checkpoint['classes']
        self.hidden_dims = checkpoint['hidden_dims']
        self.dropout_rate = checkpoint['dropout_rate']
        self.use_batch_norm = checkpoint['use_batch_norm']
        self.training_history_ = checkpoint.get('training_history', {})

        self.model = ResumeClassifierNet(
            input_dim=checkpoint['input_dim'],
            num_classes=checkpoint['num_classes'],
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_fitted_ = True

        return self


def train_neural_model(
    X_train: Union[np.ndarray, scipy.sparse.spmatrix],
    y_train: np.ndarray,
    X_val: Optional[Union[np.ndarray, scipy.sparse.spmatrix]] = None,
    y_val: Optional[np.ndarray] = None,
    hidden_dims: List[int] = None,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: Optional[str] = None,
    verbose: bool = True,
    random_state: int = 42
) -> Tuple[NeuralClassifier, Dict[str, List[float]]]:
    """
    Train a neural network classifier.

    Convenience function for training with common parameters.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        hidden_dims: Hidden layer dimensions
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use (auto-detect if None)
        verbose: Print training progress
        random_state: Random seed

    Returns:
        Tuple of (fitted classifier, training history)
    """
    classifier = NeuralClassifier(
        hidden_dims=hidden_dims,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        verbose=verbose,
        random_state=random_state
    )

    classifier.fit(X_train, y_train, X_val, y_val)

    return classifier, classifier.training_history_


# Export check for torch availability
def check_neural_model_available() -> bool:
    """Check if neural model can be used."""
    return TORCH_AVAILABLE
