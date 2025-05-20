import pandas as pd
import numpy as np
import hdbscan
import umap
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_recall_fscore_support
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import importlib
import sdo_ctu13new  # 匯入 sdo.py 中的 SDOclust
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
importlib.reload(sdo_ctu13new)  # 確保正確載入

# =================== 1. 讀取對比學習後的嵌入表示 ===================
# embedding_file = './ctu13 contrastive/ctu13_density_simple_embed.csv'
embedding_file = './ctu13 contrastive/ctu13_contrastive.csv'
# embedding_file = './ctu13 contrastive/ctu13_cleaned_resampled.csv'
df_emb = pd.read_csv(embedding_file)

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
# df_emb['Label'] = df_emb['Label'].map(label_map)
# # print(df_emb['Label'].value_counts())

print(f"嵌入資料形狀: {df_emb.shape}")
X_emb = df_emb.iloc[:, :-1].values  # 嵌入表示
y_labels = df_emb['Label'].values    # 真實標籤

# 標準化數據
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_emb)

print(f"Dataset shape: {X_emb.shape}")

for i in range(120, 601,20):
    print("-------------------------\n")
    print(f"min_cluster_size = {i}")

    noise = 0

    # ======= 2. 使用 SDOclust 對全資料進行粗分群 ======
    start_time = time.time()
    sdo_clusterer = sdo_ctu13new.SDOclust(k=i)
    sdo_clusterer.fit(X_scaled)
    sdo_labels = sdo_clusterer.predict(X_scaled)
    # sdo_clusterer.fit(X_emb)
    # sdo_labels = sdo_clusterer.predict(X_emb)

    # 計算 SDOclust 分群的時間
    sdo_execution_time = time.time() - start_time
    print(f"SDOclust execution time: {sdo_execution_time:.2f} seconds")

    # 找出不同的 SDO 群
    unique_sdo_clusters = np.unique(sdo_labels)
    print(f"SDOclust generated clusters: {unique_sdo_clusters}")

    # 初始化最終聚類標籤，預設噪聲為 -1
    final_cluster_labels = np.full(sdo_labels.shape, -1)

    # 開始處理每個 SDO 群，對每個群做 UMAP + HDBSCAN
    total_cluster_processing_time = 0  # 紀錄所有群的處理時間
    cluster_offset = 0  # 用以偏移不同 SDO 群內部 HDBSCAN 的標籤

    # 對每個 SDO 群依序處理
    for cluster in unique_sdo_clusters:
        # 取出該 SDO 群的原始嵌入資料（未經降維的 X_emb）
        indices = np.where(sdo_labels == cluster)[0]
        cluster_data = X_emb[indices]
        # cluster_data = X_scaled[indices]

        # print(f"Processing SDO cluster {cluster} with {len(cluster_data)} samples...")

        cluster_start = time.time()

        # # 使用 UMAP 將該 SDO 群的資料降維到 n_components 維
        # # umap_reducer = umap.UMAP(n_components=10, n_neighbors=10, min_dist=0.5, metric='cosine')
        # umap_reducer = umap.UMAP(n_components=20, n_neighbors=100, min_dist=0.7, metric='cosine')
        # cluster_data_reduced = umap_reducer.fit_transform(cluster_data)
        
        # # 對 UMAP 降維結果進行標準化
        # scaler = StandardScaler()
        # cluster_data_reduced_scaled = scaler.fit_transform(cluster_data_reduced)
        
        # 使用 HDBSCAN 對降維後的資料進行聚類，參數可調整
        hdb_clusterer = hdbscan.HDBSCAN(min_cluster_size=260)
        # hdb_labels = hdb_clusterer.fit_predict(cluster_data_reduced)
        hdb_labels = hdb_clusterer.fit_predict(cluster_data)
        
        # 檢查聚類結果
        print(f"SDO cluster {cluster}, HDBSCAN labels: {np.unique(hdb_labels)}")
        # print(f"Noise points: {np.sum(hdb_labels == -1)}")

        noise += np.sum(hdb_labels == -1)

        # 將該 SDO 群的 HDBSCAN 結果保存到 final_cluster_labels，
        # 噪聲保持 -1，其它標籤加上 cluster_offset 以避免重疊
        final_cluster_labels[indices] = np.where(hdb_labels == -1, -1, hdb_labels + cluster_offset)

        # 更新偏移量：根據該群非噪聲標籤最大值更新
        non_noise_labels = hdb_labels[hdb_labels != -1]
        if len(non_noise_labels) > 0:
            cluster_offset += (non_noise_labels.max() + 1)
        else:
            print(f"only noise points.")

        cluster_time = time.time() - cluster_start
        total_cluster_processing_time += cluster_time
        # print(f"SDO cluster {cluster} processing time: {cluster_time:.2f} seconds")

    # 驗證最終標籤的噪聲點數量
    # print(f"Final cluster labels (before filtering): {np.unique(final_cluster_labels)}")
    print(f"噪聲總數：", noise)

    # 計算總執行時間
    total_time = sdo_execution_time + total_cluster_processing_time
    print(f"Total execution time: {total_time:.2f} seconds")
    
    # =================== 4. 定義計算評估指標的函式 ===================
    def calculate_metrics(true_labels, cluster_labels):
        valid = cluster_labels != -1
        y_true = np.array(true_labels)[valid]
        y_clust = np.array(cluster_labels)[valid]

        # 轉成數字
        le = LabelEncoder().fit(y_true)
        y_true_num = le.transform(y_true)
        # 用匈牙利算法建混淆矩陣
        n_true = y_true_num.max() + 1
        n_clust = y_clust.max() + 1
        cm = np.zeros((n_clust, n_true), dtype=int)
        for cl, t in zip(y_clust, y_true_num):
            cm[cl, t] += 1
        row, col = linear_sum_assignment(cm.max() - cm)
        mapping = {r: c for r, c in zip(row, col)}

        # 生成预测
        y_pred = np.full_like(y_true_num, -1)
        for i, cl in enumerate(y_clust):
            if cl in mapping:
                y_pred[i] = mapping[cl]

        # 整體 Accuracy
        overall_acc = (y_pred == y_true_num).sum() / len(y_true_num)

        # 每類 Precision, Recall, F1
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true_num, y_pred, labels=list(range(n_true)), zero_division=0
        )
        class_names = le.inverse_transform(np.arange(n_true))

        # 宏平均 F1
        overall_macro_f1 = f1.mean()

        # 組字典
        per_class_precision = dict(zip(class_names, prec))
        per_class_recall    = dict(zip(class_names, rec))
        per_class_f1        = dict(zip(class_names, f1))

        return overall_acc, overall_macro_f1, per_class_precision, per_class_recall, per_class_f1

    # 呼叫並打印
    overall_acc, overall_macro_f1, per_class_prec, per_class_rec, per_class_f1 = calculate_metrics(
        y_labels, final_cluster_labels
    )
    print(f"整體聚類 Accuracy (不含噪聲): {overall_acc:.2f}")
    print(f"整體（宏平均） F1‑score   : {overall_macro_f1:.2f}\n")

    print("Class / Precision / Recall / F1‑score")
    for lbl in per_class_prec:
        p = per_class_prec[lbl]
        r = per_class_rec[lbl]
        f = per_class_f1[lbl]
        print(f"{lbl} {p:9.2f} {r:8.2f} {f:9.2f}")

    # 在下一次 k 循環前，重置
    final_cluster_labels = np.full(sdo_labels.shape, -1)
    cluster_offset = 0
