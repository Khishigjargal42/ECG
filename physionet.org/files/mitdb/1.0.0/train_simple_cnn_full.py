import os, glob, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import itertools
import json

SEG_DIR = os.path.join(os.getcwd(), "mitdb_segments")
OUT_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# keep same simplified labels as segmentation script
KEEP = ['N','V','A','F']

# load all segments
X_list = []
y_list = []
fs_list = []
for f in sorted(glob.glob(os.path.join(SEG_DIR, "*_segments.npz"))):
    data = np.load(f)
    segs = data['segments']    # (n, L)
    labs = data['labels'].astype(str)
    # filter to KEEP
    mask = np.array([lbl in KEEP for lbl in labs])
    if mask.sum() == 0:
        continue
    segs = segs[mask]
    labs = labs[mask]
    X_list.append(segs)
    y_list.append(labs)
    fs_list.append(int(data['fs']) if 'fs' in data else 360)

if len(X_list) == 0:
    raise RuntimeError("No segments found for KEEP labels. Check KEEP or mitdb_segments/")

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)
print("Total segments:", X.shape, "Labels:", np.unique(y, return_counts=True))

# encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
classes = list(le.classes_)
print("Classes:", classes)

# train/val/test split (stratified)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_enc, test_size=0.30, random_state=42, stratify=y_enc)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("Splits:", X_train.shape, X_val.shape, X_test.shape)

# dataset & dataloader
class BeatDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        x = (x - x.mean()) / (x.std()+1e-8)   # per-beat normalize
        return torch.tensor(x[None,:]), torch.tensor(self.y[idx], dtype=torch.long)

batch_size = 128
train_ds = BeatDataset(X_train, y_train)
val_ds = BeatDataset(X_val, y_val)
test_ds = BeatDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# simple 1D-CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, padding=5), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=9, padding=4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
        )
        # compute flattened size
        dummy = torch.zeros(1,1,X.shape[1])
        with torch.no_grad():
            out = self.conv(dummy)
            flat = out.view(1, -1).shape[1]
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(flat, 256), nn.ReLU(), nn.Dropout(0.4),
                                nn.Linear(256, num_classes))
    def forward(self,x): return self.fc(self.conv(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(classes)).to(device)
print("Device:", device)

# class weights
import numpy as np
counts = np.bincount(y_train)
class_weights = torch.tensor((len(y_train)/counts).astype(np.float32)).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# training loop with validation
num_epochs = 12
best_val_f1 = 0.0
history = {"train_loss":[], "val_loss":[], "val_f1":[]}

for epoch in range(1, num_epochs+1):
    model.train()
    running_loss = 0.0
    for xb,yb in train_loader:
        xb,yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*xb.size(0)
    train_loss = running_loss/len(train_ds)
    # validation
    model.eval()
    preds=[]
    trues=[]
    val_loss = 0.0
    with torch.no_grad():
        for xb,yb in val_loader:
            xb,yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            val_loss += loss.item()*xb.size(0)
            preds.append(out.argmax(dim=1).cpu().numpy())
            trues.append(yb.cpu().numpy())
    val_loss = val_loss/len(val_ds)
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    val_f1 = f1_score(trues, preds, average='weighted')
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_f1"].append(val_f1)
    print(f"Epoch {epoch}/{num_epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}")
    # save best
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))
        print("  -> saved best model")

# final evaluation on test set
model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model.pth")))
model.eval()
all_preds=[]; all_trues=[]
with torch.no_grad():
    for xb,yb in test_loader:
        xb = xb.to(device)
        out = model(xb).argmax(dim=1).cpu().numpy()
        all_preds.append(out)
        all_trues.append(yb.numpy())
all_preds = np.concatenate(all_preds)
all_trues = np.concatenate(all_trues)

# classification report & confusion matrix
report = classification_report(all_trues, all_preds, target_names=classes, output_dict=True)
df_report = pd.DataFrame(report).T
df_report.to_csv(os.path.join(OUT_DIR, "classification_report.csv"))
cm = confusion_matrix(all_trues, all_preds)
np.save(os.path.join(OUT_DIR, "confusion_matrix.npy"), cm)

# plot confusion matrix
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.
for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i,j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i,j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=200)

# save history & label encoder classes
with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
    json.dump(history, f, indent=2)
with open(os.path.join(OUT_DIR, "label_classes.json"), "w") as f:
    json.dump({"classes": classes}, f, indent=2)

print("Done. Results saved to:", OUT_DIR)
