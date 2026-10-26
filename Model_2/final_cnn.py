from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict



PROJECT = Path(__file__).parent
FEATURES = PROJECT / "features"
CLASSES = ["good", "broken", "heavy load"]

MAX_EPOCHS = 100
PATIENCE = 7
MIN_IMPROVEMENT = 1e-4


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load in features
X = np.load(f"{FEATURES}/X_train.npy").astype("float32")
y = np.load(f"{FEATURES}/y_train.npy").astype("int64")
X_test = np.load(f"{FEATURES}/X_test.npy").astype("float32")
y_test = np.load(f"{FEATURES}/y_test.npy").astype("int64")
cond_test = np.load(f"{FEATURES}/cond_test.npy")

def temporal_split(labels, val_fraction=0.2):
    train_idx, val_idx = [], []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        train_idx.extend(idx[:-n_val])   # tidlige segmenter til trening
        val_idx.extend(idx[-n_val:])     # siste segmenter til validering
    return np.array(train_idx), np.array(val_idx)

train_idx, val_idx = temporal_split(y)
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

mean, std = X_train.mean(), X_train.std()
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# Kanal-dimensjon: (N, 64, 259) -> (N, 1, 64, 259)
X_train = X_train[:, None, :, :]
X_val = X_val[:, None, :, :]
X_test = X_test[:, None, :, :]

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=16,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
)


val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
    batch_size=32,
)

test_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
    batch_size=32,
)


class EngineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),

            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.network(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = EngineCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

best_val_loss = float("inf")
patience_counter = 0

for epoch in range(MAX_EPOCHS):
    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(yb)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            val_loss += loss.item() * len(yb)

    val_loss /= len(val_loader.dataset)

    print(
        f"Epoch {epoch + 1:02d} | "
        f"train loss {train_loss:.4f} | "
        f"val loss {val_loss:.4f}"
    )

    if val_loss < best_val_loss - MIN_IMPROVEMENT:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(
            {"state_dict": model.state_dict(), "mean": float(mean), "std": float(std)},
            PROJECT / "engine_cnn.pth",
        )
    else:
        patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping after {epoch + 1} epochs.")
            break

checkpoint = torch.load(PROJECT / "engine_cnn.pth")
model.load_state_dict(checkpoint["state_dict"])
model.eval()

predictions = []

with torch.no_grad():
    for xb, _ in test_loader:
        predictions.extend(model(xb.to(device)).argmax(1).cpu().numpy())


predictions = np.array(predictions)
by_condition = defaultdict(list)
for cond, actual, pred in zip(cond_test, y_test, predictions):
    by_condition[str(cond)].append(actual == pred)

print("\nTest accuracy per støyforhold:")
for cond, results in sorted(by_condition.items()):
    print(f"{cond:15s} {np.mean(results):.3f} ({sum(results)}/{len(results)})")        

print("\nTest results:")
print(classification_report(y_test, predictions, target_names=CLASSES))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))