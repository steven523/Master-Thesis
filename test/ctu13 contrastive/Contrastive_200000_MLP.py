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

# 依照先前步驟，將特徵欄位與標籤分開
feature_cols = df.columns.drop('Label')
X = df[feature_cols].values
y_str = df['Label'].values

# 將文字標籤轉成數值ID
class_names = sorted(np.unique(y_str))  # 例如 ['BENIGN', 'Botnet - Attempted', ...]
class_to_idx = {c: i for i, c in enumerate(class_names)}
y_num = np.array([class_to_idx[label] for label in y_str], dtype=np.int64)

# 轉換成數值ID後，查看各類別數量
label_counts = pd.Series(y_num).value_counts()
print("各類別數值ID的分布：")
print(label_counts)

# 若想查看原始文字標籤對應數值，可印出 mapping
print("\n文字標籤對應的數值ID：")
print(class_to_idx)

# 標準化特徵
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 轉成 Tensor
X_tensor = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(y_num, dtype=torch.long)

# 建立 Dataset
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = TabularDataset(X_tensor, y_tensor)

# =============== 2. 定義每個類別在 batch 中抽取的數量 ===============
# 以下純範例，可依你需求做調整：
# 假設順序與 class_names 對應，或可直接用 dict[class_id] = ...
# 例如：12 個類別
# BENIGN(0)->192, Portscan(1)->16, DoS Hulk(2)->16, ... 直到最小類別2筆
class_batch_counts = {
    0: 110,    # benign     110000
    1: 14,   # menti       14395
    2: 8,  # murlo        8000
    3: 5,   # neris        5000
    4: 5,   # nsis.ay      1000
    5: 14,   # rbot        14320
    6: 5,   # virut        2000
}
# 共236

# =============== 3. 自訂 ProportionalBatchSampler ===============
class ProportionalBatchSampler(Sampler):
    """
    每個 batch 依照 class_batch_counts 設定，對每類別 with replacement 隨機抽指定數量。
    num_batches_per_epoch: 每個 epoch 產生多少個 batch
    """
    def __init__(self, labels, class_batch_counts, num_batches_per_epoch=150):
        self.labels = labels
        self.class_batch_counts = class_batch_counts
        self.num_batches_per_epoch = num_batches_per_epoch

        # 先把每個類別的所有索引存起來
        self.label_to_indices = {}
        unique_labels = torch.unique(self.labels).tolist()  # e.g. [0,1,2,...,11]
        for lab in unique_labels:
            idxs = torch.where(self.labels == lab)[0].tolist()
            self.label_to_indices[lab] = idxs

        # 計算單一batch的總大小
        self.batch_size = sum(class_batch_counts.values())

    def __iter__(self):
        for _ in range(self.num_batches_per_epoch):
            batch_indices = []
            # 針對每個類別，with replacement 隨機取出指定數量
            for lab, count in self.class_batch_counts.items():
                candidates = self.label_to_indices[lab]
                # 若某類別資料不足count, 也能重複取樣
                chosen = np.random.choice(candidates, size=count, replace=True)
                batch_indices.extend(chosen)

            # 打亂 batch 內的索引
            np.random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        # 只是約定俗成回傳 "每個 epoch 會產生多少 batch"
        return self.num_batches_per_epoch

# 使用此 Sampler
num_batches_per_epoch = 200  # 可依需求調整，每個 epoch 產生100個 batch
sampler = ProportionalBatchSampler(y_tensor, class_batch_counts, num_batches_per_epoch)
dataloader = DataLoader(dataset, batch_sampler=sampler)

print("Single batch size =", sum(class_batch_counts.values()))
print("Batches per epoch =", len(dataloader))

# =============== 4. 定義簡易 MLP 編碼器 (Encoder) ===============
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
    def forward(self, x):
        return self.net(x)

# =============== 5. Supervised Contrastive Loss (同前) ===============
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        features_norm = nn.functional.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features_norm, features_norm.t())
        labels = labels.contiguous().view(-1, 1)
        same_class_mask = torch.eq(labels, labels.t()).float().to(device)
        logits = similarity_matrix / self.temperature
        exp_logits = torch.exp(logits)
        log_prob = []
        for i in range(batch_size):
            mask_i = same_class_mask[i].clone()
            mask_i[i] = 0  # exclude self
            denominator = torch.sum(exp_logits[i] * (1 - torch.eye(batch_size, device=device)[i]))
            numerator = torch.sum(exp_logits[i] * mask_i)
            if numerator == 0:
                continue
            log_prob_i = torch.log(numerator / denominator)
            log_prob.append(log_prob_i)
        if len(log_prob) == 0:
            return torch.tensor(0.0, requires_grad=True, device=device)
        loss = - torch.mean(torch.stack(log_prob))
        return loss

# =============== 6. 建立訓練流程 ===============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X.shape[1]
encoder = MLPEncoder(input_dim, embed_dim=32).to(device)
criterion = SupConLoss(temperature=0.07)
optimizer = optim.Adam(encoder.parameters(), lr=1e-3)

epochs = 50  # 範例跑10個epoch
for epoch in range(epochs):
    encoder.train()
    total_loss = 0.0
    count_samples = 0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        features = encoder(batch_X)
        loss = criterion(features, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累計
        bs = batch_X.size(0)
        total_loss += loss.item() * bs
        count_samples += bs

    avg_loss = total_loss / count_samples
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

# =============== 7. 獲得最終嵌入 & 分群 ===============
encoder.eval()
with torch.no_grad():
    all_features = encoder(X_tensor.to(device)).cpu().numpy()
print("最終嵌入維度:", all_features.shape)

# 為嵌入建立欄位名稱，例如 emb_1, emb_2, ..., emb_64
embed_cols = [f"emb_{i+1}" for i in range(all_features.shape[1])]

# 建立 DataFrame 並加入 Label 欄位
df_emb = pd.DataFrame(all_features, columns=embed_cols)
df_emb['Label'] = df['Label'].values

# 輸出 CSV，最終會有 65 欄，前64欄為嵌入，最後1欄為 Label
output_file = "./ctu13 contrastive/ctu13_contrative_mlp.csv"
df_emb.to_csv(output_file, index=False)
print(f"已將嵌入與 Label 儲存至 {output_file}")