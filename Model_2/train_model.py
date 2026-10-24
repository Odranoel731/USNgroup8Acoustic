import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

FEATURES = "features"
CLASSES = ["good", "broken", "heavy load"]

MAX_EPOCHS = 100
PATIENCE = 7
MIN_IMPROVEMENT = 1e-4

X = np.load(f"{FEATURES}/X_train.npy")
y = np.load(f"{FEATURES}/y_train.npy")
X_test = np.load(f"{FEATURES}/X_test.npy")
y_test = np.load(f"{FEATURES}/y_test.npy")

# Add channel dimension: (N, 64, 259) -> (N, 1, 64, 259)
X = X[:, None, :, :].astype("float32")
X_test = X_test[:, None, :, :].astype("float32")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=16,
    shuffle=True,
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
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.network(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = EngineCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
        torch.save(model.state_dict(), "engine_cnn.pth")
    else:
        patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping after {epoch + 1} epochs.")
            break

model.load_state_dict(torch.load("engine_cnn.pth"))
model.eval()

predictions = []

with torch.no_grad():
    for xb, _ in test_loader:
        predictions.extend(model(xb.to(device)).argmax(1).cpu().numpy())

print("\nTest results:")
print(classification_report(y_test, predictions, target_names=CLASSES))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))