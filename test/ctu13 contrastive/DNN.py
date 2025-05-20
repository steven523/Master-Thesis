import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset

# --- 1. 載入資料 (嵌入或原始特徵) ---
# df = pd.read_csv("./ctu13 contrastive/ctu13_density_simple_embed.csv") 
# df = pd.read_csv("./ctu13 contrastive/ctu13_contrastive.csv")
df = pd.read_csv("./ctu13 contrastive/ctu13_cleaned_resampled.csv")

# 建立數字→文字的映射
label_map = {
    0: 'benign',
    1: 'neris',
    2: 'rbot',
    3: 'virut',
    4: 'menti',
    5: 'murlo',
    6: 'nsis.ay'
}

#  把原始的數字 Label 欄位替換成文字
df['Label'] = df['Label'].map(label_map)

# feature_cols = [c for c in df.columns if c.startswith('emb_')]  # 或換成原始特徵欄位列表
feature_cols = ['SrcToDst_NumOfPkts','SrcToDst_NumOfBytes','SrcToDst_Byte_Max','SrcToDst_Byte_Min','SrcToDst_Byte_Mean','DstToSrc_NumOfPkts',
                'DstToSrc_NumOfBytes','DstToSrc_Byte_Max','DstToSrc_Byte_Min','DstToSrc_Byte_Mean','Total_NumOfPkts','Total_NumOfBytes','Total_Byte_Max',
                'Total_Byte_Min','Total_Byte_Mean','Total_Byte_STD','Total_PktsRate','Total_BytesRate','Total_BytesTransferRatio','Duration']
X = df[feature_cols].values
y = LabelEncoder().fit_transform(df['Label'].values)

print(df['Label'].value_counts())
print("資料形狀:", df.shape)

# --- 2. Dataset & DataLoader ---
class FlowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = FlowDataset(X, y)
# n_train = int(len(dataset) * 0.8)
# train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train])

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(X, y))

train_ds = Subset(dataset, train_idx)
val_ds   = Subset(dataset, val_idx)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)

# --- 3. 定義 4 層 DNN 分類器架構 ---
class DNNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128,  64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64,  32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# --- 4. 初始化模型、損失與優化器 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DNNClassifier(input_dim=X.shape[1], num_classes=len(np.unique(y)), dropout=0.3)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

# --- 5. 訓練與驗證迴圈 ---
for epoch in range(1, 21):
    # ——— 训练 ———
    model.train()
    train_loss, train_acc, n_train = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        preds = logits.argmax(dim=1)
        train_loss += loss.item() * xb.size(0)
        train_acc  += (preds == yb).sum().item()
        n_train    += xb.size(0)

    train_loss /= n_train
    train_acc  = train_acc / n_train

    # ——— 验证 ———
    model.eval()
    val_loss, val_acc, n_val = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(dim=1)
            val_loss += loss.item() * xb.size(0)
            val_acc  += (preds == yb).sum().item()
            n_val    += xb.size(0)

            # 收集用于计算 precision/recall/f1
            all_preds .extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    val_loss /= n_val
    val_acc   = val_acc / n_val

    # 计算额外指标（macro 平均）
    val_prec  = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    val_rec   = recall_score   (all_labels, all_preds, average='macro', zero_division=0)
    val_f1    = f1_score       (all_labels, all_preds, average='macro', zero_division=0)

    # ——— 打印汇总 ———
    print(
        f"Epoch {epoch:2d} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val Prec: {val_prec:.4f} | "
        f"Val Rec: {val_rec:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )

