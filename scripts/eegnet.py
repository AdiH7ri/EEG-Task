import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import json

from config import ExpParams

class EEGNet(nn.Module):
    """
    PyTorch implementation of EEGNet (Lawhern et al., 2018)
    """
    def __init__(self, n_channels=32, n_samples=256, n_classes=2, dropout=0.5):
        super(EEGNet, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )
        self._n_flatten = 32 * (n_samples // 32)
        self.classify = nn.Linear(self._n_flatten, n_classes)

    def forward(self, x):
        # Input shape: (batch, channels, samples)
        x = x.unsqueeze(1)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.flatten(start_dim=1)
        return self.classify(x)

class EEGDataset(Dataset):
    """
    Simple dataset wrapper for EEG epochs.
    X: (n_epochs, n_channels, n_timesteps)
    y: (n_epochs,)
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(model, train_loader, test_loader, device, n_epochs=30, lr=1e-3) -> tuple:
    '''
    Train the EEGNet model.
    
    Args:
        model: EEGNet instance
        train_loader: DataLoader for training data
        test_loader: DataLoader for validation data
        device: torch device (cpu or cuda)
        n_epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        train_losses: List of training losses per epoch
        test_losses: List of validation losses per epoch
        test_accs: List of validation accuracies per epoch
    '''
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses, test_accs = [], [], []

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Validation
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                test_loss += loss.item()
                pred = out.argmax(1)
                correct += (pred == yb).sum().item()
        acc = correct / len(test_loader.dataset)
        test_losses.append(test_loss / len(test_loader))
        test_accs.append(acc)

    return train_losses, test_losses, test_accs


def cross_validate(
    model_class, dataset, save_dir, n_splits=5, n_epochs=30, lr=1e-3, device=None, test_ratio=0.2
) -> tuple:
    """
    n-fold cross-validation with an external held-out test set.
    The best fold model is evaluated on the held-out test set.

    Args:
        model_class: Callable that returns a new instance of the model
        dataset: EEGDataset instance
        save_dir: Directory to save artifacts
        n_splits: Number of CV folds
        n_epochs: Number of training epochs
        lr: Learning rate
        device: torch device (cpu or cuda)
        test_ratio: Proportion of data to hold out as final test set
    
    Returns:
        summary_df: DataFrame summarizing CV results
        test_metrics: Dictionary with held-out test metrics
        
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Split into training+validation and held-out test set 
    total_len = len(dataset)
    test_len = int(total_len * test_ratio)
    train_val_len = total_len - test_len
    train_val_set, test_set = random_split(dataset, [train_val_len, test_len])
    print(f"Data split: {train_val_len} (train+val) | {test_len} (held-out test)")

    y_train_val = np.array([dataset[i][1] for i in train_val_set.indices])

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=ExpParams.SEED)
    results = []
    best_model_path, best_val_acc = None, -1.0

    # Perform cross-validation on training+validation set 
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(train_val_set)), y_train_val)):
        print(f"\n===== Fold {fold+1}/{n_splits} =====")
        train_sub = Subset(train_val_set, train_idx)
        val_sub = Subset(train_val_set, val_idx)

        train_loader = DataLoader(train_sub, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=64, shuffle=False)

        model = model_class().to(device)
        train_losses, val_losses, val_accs = train_model(model, train_loader, val_loader, device, n_epochs, lr)

        fold_dir = save_dir / f"fold_{fold+1}"
        fold_dir.mkdir(exist_ok=True)

        # Save fold metrics as CSV
        df = pd.DataFrame({
            "epoch": np.arange(1, n_epochs + 1),
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_acc": val_accs
        })
        df.to_csv(fold_dir / "fold_metrics.csv", index=False)

        # Save model
        model_path = fold_dir / "eegnet_weights.pt"
        torch.save(model.state_dict(), model_path)

        # Plot curves
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Val")
        plt.title(f"Fold {fold+1} - Loss")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(val_accs, label="Val Accuracy")
        plt.title(f"Fold {fold+1} - Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fold_dir / "training_curves.png")
        plt.close()

        final_acc = val_accs[-1]
        results.append({"fold": fold+1, "final_val_acc": final_acc, "min_val_loss": min(val_losses)})

        if final_acc > best_val_acc:
            best_val_acc = final_acc
            best_model_path = model_path

    # Summarize folds 
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(save_dir / "cv_summary.csv", index=False)
    print(f"\n Cross-validation complete. Mean val acc: {summary_df.final_val_acc.mean():.3f}")
    print(f" Best model from Fold {np.argmax(summary_df.final_val_acc.values) + 1} with Val Acc = {best_val_acc:.3f}")

    # Evaluate best model on held-out test set 
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    best_model = model_class().to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = best_model(xb)
            loss = criterion(out, yb)
            test_loss += loss.item()
            pred = out.argmax(1)
            correct += (pred == yb).sum().item()

    test_acc = correct / len(test_loader.dataset)
    test_loss /= len(test_loader)
    print(f"\n Held-out Test Accuracy: {test_acc:.3f} | Test Loss: {test_loss:.4f}")

    #  Save final test metrics 
    test_metrics = {
        "best_fold": int(np.argmax(summary_df.final_val_acc.values) + 1),
        "val_acc_best": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss)
    }
    with open(save_dir / "final_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    return summary_df, test_metrics