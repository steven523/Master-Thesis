import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.preprocessing import StandardScaler

# =============== 1. 讀取資料與前處理 ===============
df = pd.read_csv('./ctu13 contrastive/ctu13_cleaned_resampled.csv')

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
print(df['Label'].value_counts())

print("shape: ",df.shape)

feature_cols = df.columns.drop('Label')
X = df[feature_cols].values
y_str = df['Label'].values

class_names = sorted(np.unique(y_str))
class_to_idx = {c: i for i, c in enumerate(class_names)}
y_num = np.array([class_to_idx[label] for label in y_str], dtype=np.int64)

print("各類別數值ID的分布：")
print(pd.Series(y_num).value_counts())
print("\n文字標籤對應的數值ID：")
print(class_to_idx)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_tensor = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(y_num, dtype=torch.long)

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X; self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = TabularDataset(X_tensor, y_tensor)

# =============== 2. 定義每個類別在 batch 中抽取的數量 ===============

class_batch_counts = {
    0: 110,    # benign     160000
    1: 14,   # menti       14395
    2: 8,  # murlo        8000
    3: 5,   # neris        5000
    4: 5,   # nsis.ay      1000
    5: 14,   # rbot        14320
    6: 5,   # virut        2000
}
# 共236

num_batches_per_epoch = 150

class ProportionalBatchSampler(Sampler):
    def __init__(self, labels, class_batch_counts, num_batches_per_epoch):
        self.labels = labels
        self.class_batch_counts = class_batch_counts
        self.num_batches_per_epoch = num_batches_per_epoch
        self.label_to_indices = {
            lab: torch.where(labels==lab)[0].tolist()
            for lab in torch.unique(labels).tolist()
        }
        self.batch_size = sum(class_batch_counts.values())
    def __iter__(self):
        for _ in range(self.num_batches_per_epoch):
            batch = []
            for lab, cnt in self.class_batch_counts.items():
                idxs = self.label_to_indices[lab]
                batch.extend(np.random.choice(idxs, size=cnt, replace=True))
            np.random.shuffle(batch)
            yield batch
    def __len__(self):
        return self.num_batches_per_epoch

sampler = ProportionalBatchSampler(y_tensor, class_batch_counts, num_batches_per_epoch)
dataloader = DataLoader(dataset, batch_sampler=sampler)
print("Single batch size =", sum(class_batch_counts.values()))
print("Batches per epoch =", len(dataloader))

# =============== 4. 定義 ResMLPEncoder ===============
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),          # 改成非就地
            nn.Linear(dim, dim)
        )
        self.act = nn.ReLU()    # 改成非就地

    def forward(self, x):
        return self.act(x + self.net(x))

class ResMLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, embed_dim=32, num_blocks=10):
        super().__init__()
        # 輸入升維
        self.input_lin = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # N 個殘差塊
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        # 投影到嵌入
        self.project = nn.Linear(hidden_dim, embed_dim)
    def forward(self, x):
        x = self.input_lin(x)
        x = self.blocks(x)
        return self.project(x)

# 總層數計算：
#   1 input Linear
# + 10 × (2 Linear per block)
# + 1 output Linear
# = 1 + 20 + 1 = 22 層 Linear
# 加上 ReLU 後總層數也就是 22 個全連接層 + 若干個 ReLU

# =============== 5. SupConLoss ===============
import torch
import torch.nn.functional as F
from torch import nn

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, features, labels):
        """
        features:   [B, D] 未归一化的 embedding
        labels:     [B]   int64
        """
        device = features.device
        batch_size = features.shape[0]
        if batch_size == 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        # 1) L2 归一化
        f = F.normalize(features, dim=1)            # [B, D]
        # 2) 相似度矩阵
        sim = torch.matmul(f, f.t()) / self.t        # [B, B]
        # 3) 正样本掩码
        labels = labels.view(-1,1)
        mask  = torch.eq(labels, labels.t()).float().to(device)  # [B, B]
        # 把对角（自相似）设为 0
        mask = mask - torch.eye(batch_size, device=device)
        # 4) 指数化
        exp_sim = torch.exp(sim)                     # [B, B]
        # 5) 分子：每行上，所有正样本的位置求和
        numerator = (exp_sim * mask).sum(dim=1)      # [B]
        # 6) 分母：每行上，除去 self 的所有位置求和
        denom = (exp_sim * (1 - torch.eye(batch_size, device=device))).sum(dim=1)  # [B]
        # 7) 搞成对比 loss
        # 为了避免分子为 0 导致 log(nan)，我们先掐掉那些行
        valid = numerator > 0
        loss  = -torch.log(numerator[valid] / denom[valid])
        # 8) 平均
        return loss.mean() if valid.any() else torch.tensor(0.0, device=device, requires_grad=True)

# =============== 6. 訓練 ===============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = ResMLPEncoder(input_dim=X.shape[1], hidden_dim=128, embed_dim=32, num_blocks=10).to(device)
criterion = SupConLoss(temperature=0.03).to(device)
optimizer = optim.Adam(encoder.parameters(), lr=1e-3)

epochs = 50
for ep in range(epochs):
    encoder.train()
    tot_loss = 0.0; cnt=0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        feats = encoder(batch_X)
        loss = criterion(feats, batch_y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        bs = batch_X.size(0)
        tot_loss += loss.item()*bs; cnt += bs
    print(f"Epoch [{ep+1}/{epochs}] - Loss: {tot_loss/cnt:.4f}")

# =============== 7. 獲得嵌入 ===============
encoder.eval()
with torch.no_grad():
    emb = encoder(X_tensor.to(device)).cpu().numpy()
print("最終嵌入維度:", emb.shape)

cols = [f"emb_{i+1}" for i in range(emb.shape[1])]
df_out = pd.DataFrame(emb, columns=cols)
df_out['Label'] = df['Label'].values
df_out.to_csv("./ctu13 contrastive/ctu13_contrastive.csv", index=False)
print("已將嵌入與 Label 儲存完成。")
