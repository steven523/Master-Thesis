import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F

# =============== 1. 讀取資料與前處理 ===============
df = pd.read_csv('./ctu13 contrastive/ctu13_cleaned_resampled.csv')
label_map = {0:'benign',1:'neris',2:'rbot',3:'virut',4:'menti',5:'murlo',6:'nsis.ay'}
df['Label'] = df['Label'].map(label_map)
feature_cols = df.columns.drop('Label')
X = df[feature_cols].values
y_str = df['Label'].values
class_names = sorted(np.unique(y_str))
class_to_idx = {c:i for i,c in enumerate(class_names)}
y = np.array([class_to_idx[l] for l in y_str], dtype=np.int64)

print("各類別數值ID的分布：")
print(pd.Series(y).value_counts())
print("\n文字標籤對應的數值ID：")
print(class_to_idx)

# 標準化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =============== 2. kNN 距離與类级閾值切分 ===============
k = 20  # 使用20个邻居
nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
dists, idxs = nbrs.kneighbors(X)
avg_knn = dists[:,1:].mean(axis=1)
# 類別級別的中位數閾值
class_median = {cls: np.median(avg_knn[y==cls]) for cls in np.unique(y)}
# 每個樣本根據自身類別閾值判定是否緊密
thr = np.array([class_median[y[i]] for i in range(len(X))])
is_dense = avg_knn < thr  # True 表示紧密，False 表示稀疏
# 計算異類鄰居比例 R
neighbor_idxs = idxs[:,1:]
R = np.array([(y[neighbor_idxs[i]] != y[i]).sum() / k for i in range(len(X))])

# =============== 3. Dataset 与 Sampler ===============
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx

dataset = TabularDataset(X, y)
class ProportionalBatchSampler(Sampler):
    def __init__(self, labels, counts, num_batches):
        self.labels = labels
        self.counts = counts
        self.num_batches = num_batches
        self.idx_map = {int(lab.item()): torch.where(labels==lab)[0].tolist() for lab in torch.unique(labels)}
    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for lab,cnt in self.counts.items(): batch += list(np.random.choice(self.idx_map[lab], cnt, replace=True))
            np.random.shuffle(batch)
            yield batch
    def __len__(self): return self.num_batches
class_batch_counts = {
    0: 25,    # benign     160000
    1: 13,   # menti       14395
    2: 13,  # murlo        8000
    3: 15,   # neris        5000
    4: 11,   # nsis.ay      1000
    5: 13,   # rbot        14320
    6: 13,   # virut        2000
}
from torch import tensor
sampler = ProportionalBatchSampler(tensor(y), class_batch_counts, num_batches=130)
dataloader = DataLoader(dataset, batch_sampler=sampler)
print("Single batch size =", sum(class_batch_counts.values()))

# =============== 4. ResMLPEncoder ===============
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim,dim), nn.ReLU(), nn.Linear(dim,dim))
        self.act = nn.ReLU()
    def forward(self, x): return self.act(x + self.net(x))
class ResMLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, embed_dim=32, num_blocks=10):
        super().__init__()
        self.input_lin = nn.Sequential(nn.Linear(input_dim,hidden_dim), nn.ReLU())
        self.blocks    = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.project   = nn.Linear(hidden_dim, embed_dim)
    def forward(self, x):
        x = self.input_lin(x)
        x = self.blocks(x)
        return self.project(x)

# =============== 5. SupConLoss: 使用 num/(num+den) 公式 ===============
# w_pos = 2.7; 
w_pos_dict = {
    0: 1.3,   # benign
    1: 2.9,   # menti
    2: 2.5,   # murlo
    3: 7.0,   # neris
    4: 6.8,   # nsis.ay
    5: 5.5,   # rbot
    6: 6.0,   # virut
}
alpha_up, alpha_down = 0.1,0.6; 
tau_low,tau_high = 0.2,0.8
is_dense_tensor = torch.tensor(is_dense, dtype=torch.bool)
R_tensor = torch.tensor(R, dtype=torch.float32)
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07): super().__init__(); self.t=temperature
    def forward(self, feats, labels, idxs):
        B = feats.size(0)
        if B<=1: return torch.tensor(0., device=feats.device)
        f = F.normalize(feats, dim=1)
        sim = torch.matmul(f,f.t())/self.t
        exp_sim = torch.exp(sim)
        device = feats.device
        dense = is_dense_tensor.to(device)[idxs]
        Rloc  = R_tensor.to(device)[idxs]
        lab = labels.view(-1,1)
        pos_mask = torch.eq(lab, lab.t()); pos_mask.fill_diagonal_(False)
        neg_mask = ~pos_mask
        w = torch.ones_like(sim)
        # w[pos_mask] = w_pos
        labels_flat = labels.view(-1)                   # [B]
        for cls_id, wp in w_pos_dict.items():
            # 挑出 anchor i 属于 cls_id 的正对 (i,j)，并赋值
            # pos_mask[i,j] 且 labels[i]==cls_id
            cls_anchor = (labels_flat == cls_id).unsqueeze(1)   # [B,1]
            mask = pos_mask & cls_anchor                        # [B,B]
            w[mask] = wp

        sparse = ~dense; ss = sparse.unsqueeze(1)&sparse.unsqueeze(0)
        any_dense = ~ss; dn = neg_mask & any_dense
        Ri = Rloc.unsqueeze(1).expand(B,B); Rj = Rloc.unsqueeze(0).expand(B,B)
        Rp = torch.max(Ri,Rj)
        # 三档优先：同群 -> 重叠 -> 噪声
        w[dn & (Rp < tau_low)] = max(1 - alpha_down, 0.0)
        w[dn & (Rp >= tau_low) & (Rp <= tau_high)] = 1.0
        w[dn & (Rp > tau_high)] = 1 + alpha_up
        num = (exp_sim*w*pos_mask).sum(dim=1)
        den_neg = (exp_sim*w*neg_mask).sum(dim=1)
        den = num+den_neg
        valid = den>0
        loss = torch.zeros(B, device=device)
        loss[valid] = -torch.log(num[valid]/den[valid])
        return loss.mean()

# =============== 6. 訓練 ===============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = ResMLPEncoder(input_dim=X.shape[1]).to(device)
criterion=SupConLoss(temperature=0.03).to(device)
optimizer=optim.Adam(encoder.parameters(), lr=1e-3)
X_tensor=torch.tensor(X,dtype=torch.float32)
y_tensor=torch.tensor(y,dtype=torch.long)
for ep in range(1,51):
    encoder.train(); tloss,tn=0.0,0
    for Xb,yb,idxb in dataloader:
        Xb,yb,idxb = Xb.to(device), yb.to(device), idxb.to(device)
        feats=encoder(Xb); loss=criterion(feats,yb,idxb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        tloss+=loss.item()*len(Xb); tn+=len(Xb)
    print(f"Epoch {ep} — Loss: {tloss/tn:.4f}")

# =============== 7. 嵌入輸出 ===============
encoder.eval()
with torch.no_grad(): emb=encoder(X_tensor.to(device)).cpu().numpy()
df_out=pd.DataFrame(emb,columns=[f'emb_{i+1}' for i in range(emb.shape[1])])
df_out['Label']=df['Label']
df_out.to_csv('./ctu13 contrastive/ctu13_density_simple_embed_2.csv',index=False)
print('Saved embeddings.')
