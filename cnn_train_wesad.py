import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from preprocess_wesad_all_wrist_subjectwise import preprocess_wesad_dataset_subjectwise, LABEL_NAMES


# -----------------------------
# Dataset Class
# -----------------------------
class WESADDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, 320, 6)
        # For CNN we use (N, channels, seq_len)
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (N, 6, 320)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# Simple 1D CNN Model
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=6, num_classes=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # output shape: (N, 128, 1)
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.net(x)          # (N, 128, 1)
        x = x.squeeze(-1)        # (N, 128)
        x = self.classifier(x)   # (N, 4)
        return x


# -----------------------------
# Train Function
# -----------------------------
def train_model(model, train_loader, device, epochs=10, lr=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")


# -----------------------------
# Evaluate Function
# -----------------------------
@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    for Xb, yb in loader:
        Xb = Xb.to(device)
        out = model(Xb)
        pred = torch.argmax(out, dim=1).cpu().numpy()

        y_true.extend(yb.numpy())
        y_pred.extend(pred)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return acc, macro_f1, np.array(y_true), np.array(y_pred)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    WESAD_ROOT = r"C:\Users\lenovo\Desktop\minor\WESAD"

    SUBJECTS = ["S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S13","S14","S15","S16","S17"]

    WINDOW_SIZE = 320
    STEP_SIZE = 160

    # Load data subjectwise
    X, y, groups = preprocess_wesad_dataset_subjectwise(WESAD_ROOT, SUBJECTS, WINDOW_SIZE, STEP_SIZE)

    # -----------------------------
    # Simple train-test split by SUBJECTS
    # (This is easier than LOSO for CNN)
    # We'll keep last 3 subjects for testing
    # -----------------------------
    test_subjects = ["S15", "S16", "S17"]

    train_mask = ~np.isin(groups, test_subjects)
    test_mask = np.isin(groups, test_subjects)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print("\nTrain samples:", len(X_train))
    print("Test samples:", len(X_test))

    # DataLoaders
    train_ds = WESADDataset(X_train, y_train)
    test_ds = WESADDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Model
    model = SimpleCNN(in_channels=6, num_classes=4).to(device)

    # Train
    start = time.time()
    train_model(model, train_loader, device, epochs=12, lr=1e-3)
    end = time.time()
    print("\nTraining time (sec):", round(end - start, 2))

    # Evaluate
    acc, macro_f1, y_true, y_pred = evaluate_model(model, test_loader, device)

    print("\n==============================")
    print("CNN TEST RESULTS")
    print("==============================")
    print("Accuracy:", round(acc, 4))
    print("Macro F1:", round(macro_f1, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(LABEL_NAMES.values())))

    # Save model
    save_path = "cnn_wesad.pth"
    torch.save(model.state_dict(), save_path)
    print("\n✅ Saved CNN model:", save_path)
