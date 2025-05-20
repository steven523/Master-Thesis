import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# 1. 讀入 LOF 清洗並重採樣後的資料
# df = pd.read_csv("./ctu13 contrastive/ctu13_density_simple_embed.csv") 
df = pd.read_csv("./ctu13 contrastive/ctu13_contrastive.csv")
# df = pd.read_csv("./ctu13 contrastive/ctu13_cleaned_resampled.csv")

# # 建立數字→文字的映射
# label_map = {
#     0: 'benign',
#     1: 'neris',
#     2: 'rbot',
#     3: 'virut',
#     4: 'menti',
#     5: 'murlo',
#     6: 'nsis.ay'
# }

# #  把原始的數字 Label 欄位替換成文字
# df['Label'] = df['Label'].map(label_map)
# print(df['Label'].value_counts())

# 2. 準備運算
features  = [c for c in df.columns if c != "Label"]
k         = 2        # k-NN 中的 k
trim_frac = 0.10     # 截尾比例

results = []
for label, grp in df.groupby("Label"):
    X = grp[features].values
    # 標準化
    X_scaled = StandardScaler().fit_transform(X)
    # 計算 k+1 最近鄰（第 1 個是自己，故取後 k 個）
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X_scaled)
    dists, _ = nbrs.kneighbors(X_scaled)
    # 每筆樣本的 k-NN 平均距離
    mean_k = dists[:, 1:].mean(axis=1)
    # 原始平均松散度
    orig_disp = mean_k.mean()
    
    # 截尾平均松散度
    sorted_d = np.sort(mean_k)
    lo = int(len(sorted_d) * trim_frac)
    hi = len(sorted_d) - lo
    trimmed_disp = sorted_d[lo:hi].mean()
    results.append((label, orig_disp))
    # results.append((label, orig_disp, trimmed_disp))

# 3. 整理成表
df_disp = pd.DataFrame(
    results,
    columns=["Label", "Original Sparsity"]
    # columns=["Label", "Original Sparsity", "Trimmed Sparsity"]
)

print(df_disp.to_string(index=False))

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import NearestNeighbors

# # 1. 讀入 LOF 清洗並重採樣後的資料
# #    請將路徑改成你實際的檔案路徑
# # df = pd.read_csv("./ctu13 contrastive/ctu13_cleaned_resampled.csv")
# df = pd.read_csv("./ctu13 contrastive/ctu13_density_simple_embed.csv") 
# # df = pd.read_csv("./ctu13 contrastive/ctu13_contrastive.csv")

# # # 建立數字→文字的映射
# # label_map = {
# #     0: 'benign',
# #     1: 'neris',
# #     2: 'rbot',
# #     3: 'virut',
# #     4: 'menti',
# #     5: 'murlo',
# #     6: 'nsis.ay'
# # }

# # #  把原始的數字 Label 欄位替換成文字
# # df['Label'] = df['Label'].map(label_map)
# # # print(df['Label'].value_counts())

# # label_counts = df['Label'].value_counts()
# # print(f"Label 分布:")
# # print(label_counts)

# # 2. 參數設定
# features  = [c for c in df.columns if c != "Label"]
# k         = 20        # k-NN 中的 k
# trim_frac = 0.10     # 截尾比例

# # 3. 計算相對稀疏度
# records = []
# for label, grp in df.groupby("Label"):
#     X = grp[features].values
#     # 標準化
#     Xs = StandardScaler().fit_transform(X)
#     # 找 k+1 最近鄰（第0個是自己）
#     nbrs = NearestNeighbors(n_neighbors=k+1).fit(Xs)
#     dists, _ = nbrs.kneighbors(Xs)
#     mean_k = dists[:, 1:].mean(axis=1)  # 每樣本 k-NN 平均距離
    
#     # 原始
#     mu_orig = mean_k.mean()
#     # sigma_orig = mean_k.std()
#     # rel_orig = mu_orig / sigma_orig if sigma_orig>0 else np.nan
    
#     # # 截尾
#     # sorted_k = np.sort(mean_k)
#     # lo = int(len(sorted_k) * trim_frac)
#     # hi = len(sorted_k) - lo
#     # trimmed = sorted_k[lo:hi]
#     # mu_trim = trimmed.mean()
#     # sigma_trim = trimmed.std()
#     # rel_trim = mu_trim / sigma_trim if sigma_trim>0 else np.nan
    
#     records.append({
#         "Label": label,
#         # "RelativeOriginal": rel_orig,
#         "RelativeOriginal": mu_orig
#         # "RelativeTrimmed": rel_trim
#     })

# df_rel = pd.DataFrame(records)

# # 4. 格式化顯示到小數點 4 位
# df_rel["RelativeOriginal"] = df_rel["RelativeOriginal"].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "nan")
# # df_rel["RelativeTrimmed"]  = df_rel["RelativeTrimmed"].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "nan")

# print(df_rel.to_string(index=False))
