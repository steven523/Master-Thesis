import pandas as pd
import numpy as np
import hdbscan
import umap
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time
import matplotlib.pyplot as plt
import warnings
from joblib import Memory
warnings.filterwarnings("ignore", category=FutureWarning)

# =================== 1. 讀取對比學習後的嵌入表示 ===================
# embedding_file = './ctu13 contrastive/ctu13_density_simple_embed.csv'
# embedding_file = './ctu13 contrastive/ctu13_contrastive.csv'
embedding_file = './ctu13 contrastive/ctu13_cleaned_resampled.csv'
df_emb = pd.read_csv(embedding_file)

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
df_emb['Label'] = df_emb['Label'].map(label_map)
print(df_emb['Label'].value_counts())

# 假設前64欄為嵌入，最後一欄為 Label
X_emb = df_emb.iloc[:, :-1].values  # 嵌入表示
y_labels = df_emb['Label'].values    # 真實標籤
print(f"嵌入資料形狀: {df_emb.shape}")

# # 可選：若嵌入需要再標準化
# scaler = StandardScaler()
# X_emb_scaled = scaler.fit_transform(X_emb)

for i in range(180,701,20):
    print("-------------------------\n")
    print(f"min_cluster_size = {i}")

    # # 開始 UMAP 降維的計算時間
    # start_time = time.time()

    # # # 使用 UMAP 將數據從高維降低維
    # # umap_reducer = umap.UMAP(n_components=10, n_neighbors=100, min_dist=0.7, metric='cosine')
    # # umap_reducer = umap.UMAP(n_components=i, n_neighbors=10, min_dist=0.01, metric='cosine')
    # umap_reducer = umap.UMAP(n_components=i, n_neighbors=100, min_dist=0.7, metric='cosine')
    
    # # X_emb_reduced = umap_reducer.fit_transform(X_emb)
    # X_emb_reduced = umap_reducer.fit_transform(X_emb_scaled)

    # # # 標準化數據
    # # scaler = StandardScaler()
    # # X_emb_scaled = scaler.fit_transform(X_emb_reduced)

    # # 結束 UMAP 降維的計算時間
    # end_time = time.time()
    # umap_execution_time = end_time - start_time
    # print(f"UMAP execution time: {umap_execution_time:.2f} seconds")

    # =================== 2. HDBSCAN 聚類 ===================
    start_time = time.time()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=i,  # 可根據需求調整參數  
    )
    start_time = time.time()
    # cluster_labels = clusterer.fit_predict(X_emb_scaled)
    cluster_labels = clusterer.fit_predict(X_emb)
    # cluster_labels = clusterer.fit_predict(X_emb_reduced)
    end_time = time.time()
    print(f"HDBSCAN 執行時間: {end_time - start_time:.2f} seconds")

    unique_clusters = np.unique(cluster_labels)
    print(f"聚類後的群集 (包含噪聲): {unique_clusters}")
    print(f"噪聲點數量: {np.sum(cluster_labels == -1)}")

    # =================== 3. 定義計算整體 Accuracy 與各類別 Recall 的函式 ===================

    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from sklearn.preprocessing import LabelEncoder

    def calculate_metrics_full(true_labels, cluster_labels):
        """
        返回：
        - overall_accuracy      整体 Accuracy（微平均）
        - overall_macro_f1      宏平均 F₁-score
        - per_class_precision   dict: {label: precision}
        - per_class_recall      dict: {label: recall}
        - per_class_f1          dict: {label: f1}
        """
        # 1. 只保留非噪声点
        mask = cluster_labels != -1
        y_true = np.array(true_labels)[mask]
        y_clu  = np.array(cluster_labels)[mask]

        # 2. 数值化真标签
        le = LabelEncoder()
        y_true_num = le.fit_transform(y_true)
        n_true   = y_true_num.max() + 1
        n_clu    = y_clu.max() + 1

        # 3. 构造混淆矩阵 (行: cluster, 列: true)
        conf = np.zeros((n_clu, n_true), dtype=int)
        for i, c in enumerate(y_clu):
            conf[c, y_true_num[i]] += 1

        # 4. 匈牙利算法匹配 → 整体 Accuracy
        row, col = linear_sum_assignment(conf.max() - conf)
        overall_accuracy = conf[row, col].sum() / len(y_true_num)
        mapping = {r: c for r, c in zip(row, col)}

        # 5. 生成每个样本的预测标签
        y_pred = np.full_like(y_true_num, fill_value=-1)
        for i, c in enumerate(y_clu):
            if c in mapping:
                y_pred[i] = mapping[c]

        # 6. 逐类别计算 Precision, Recall, F1
        per_class_precision = {}
        per_class_recall    = {}
        per_class_f1        = {}

        for cls in range(n_true):
            tp = np.sum((y_true_num == cls) & (y_pred == cls))
            fp = np.sum((y_true_num != cls) & (y_pred == cls))
            fn = np.sum((y_true_num == cls) & (y_pred != cls))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                        if (precision + recall) > 0 else 0.0)

            label_name = le.inverse_transform([cls])[0]
            per_class_precision[label_name] = precision
            per_class_recall   [label_name] = recall
            per_class_f1       [label_name] = f1

        # 7. 宏平均 F1
        overall_macro_f1 = np.mean(list(per_class_f1.values()))

        return overall_accuracy, overall_macro_f1, per_class_precision, per_class_recall, per_class_f1

    # 在你的主循环中调用：
    overall_acc, overall_macro_f1, per_prec, per_rec, per_f1 = \
        calculate_metrics_full(y_labels, cluster_labels)

    print(f"\n整體聚類 Accuracy (不含噪聲): {overall_acc:.2f}")
    print(f"整體（宏平均）F₁-score: {overall_macro_f1:.2f}\n")

    print("各類別 Precision / Recall / F₁-score:")
    for lbl in per_prec:
        p = per_prec[lbl]
        r = per_rec [lbl]
        f = per_f1  [lbl]
        print(f"{lbl}: Precision={p:.2f}, Recall={r:.2f}, F1={f:.2f}")




