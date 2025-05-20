# import pandas as pd

# # 讀取合併後的資料集
# file_path = 'ctu13_label.pkl'
# df = pd.read_pickle(file_path)

# # 指定要刪除的特徵
# columns_to_drop = [
#     'Protocol', 'SrcIP', 'SrcPort', 'DstIP', 'DstPort'
# ]

# # 刪除指定欄位 (若欄位不存在，errors='ignore'避免拋出錯誤)
# df_reduced = df.drop(columns=columns_to_drop, errors='ignore')

# # # 刪除原本的 Label 欄位（只含 0/1）
# # if 'Label' in df_reduced.columns:
# #     df_reduced = df_reduced.drop(columns=['Label'])

# # 將 Attack 欄位改名為 Label
# df_reduced = df_reduced.rename(columns={'label': 'Label'})

# # 將處理後的資料儲存到新檔案
# output_file = './ctu13 contrastive/ctu13_clean.csv'
# df_reduced.to_csv(output_file, index=False)

# print(f"已刪除指定的特徵，並將結果儲存至 {output_file}")

# print("資料形狀:", df_reduced.shape)
# print("前二十行內容:\n", df_reduced.head(20))

# # 顯示每個檔案的 label 分布情況
# label_counts = df_reduced['Label'].value_counts()
# print(f"Label 分布:")
# print(label_counts)

# # -----------------------------------
# import pandas as pd

# df = pd.read_csv("./2017new/merged_CICIDS2017_processed.csv")

# # 定義需要刪除的 Label 類別列表
# labels_to_remove = [
#     "Dos GoldenEye",
#     "Web Attack - Brute Force - Attempted"
#     # "Botnet - Attempted",
#     # "DoS Slowloris",
#     # "DoS Slowhttptest - Attempted"
#     # "DoS Slowloris - Attempted",
#     # "DoS Slowhttptest",
#     # "Botnet",
#     # "Web Attack - XSS - Attempted",
#     # "DoS Hulk - Attempted",
#     # "DoS GoldenEye - Attempted",
#     # "Web Attack - Brute Force",
#     # "Infiltration - Attempted",
#     # "Infiltration",
#     # "SSH-Patator - Attempted",
#     # "Web Attack - XSS",
#     # "Web Attack - SQL Injection",
#     # "FTP-Patator - Attempted",
#     # "Heartbleed",
#     # "Web Attack - SQL Injection - Attempted"
# ]

# # 過濾掉 Label 為上述類別的資料
# df_filtered = df[~df['Label'].isin(labels_to_remove)]

# # 將處理後的資料儲存到新檔案
# output_file = "./2017new/merged_CICIDS2017_processed.csv"
# df_filtered.to_csv(output_file, index=False)

# print(f"已刪除指定類別，結果儲存至 {output_file}")

# -----------------------------------

# import pandas as pd
# import numpy as np

# # 讀取資料
# df = pd.read_csv("./ctu13 contrastive/ctu13_clean.csv")

# # 1. 找出哪些欄位含有 inf / -inf，並統計數量
# inf_counts = {}
# for col in df.columns:
#     if pd.api.types.is_numeric_dtype(df[col]):
#         # 建立布林陣列標示 inf 或 -inf
#         is_inf = np.isinf(df[col])
#         count = is_inf.sum()
#         if count > 0:
#             inf_counts[col] = int(count)

# # 2. 輸出結果
# if inf_counts:
#     print("以下欄位含有無限值 (inf 或 -inf)，並統計其數量：")
#     for col, cnt in inf_counts.items():
#         print(f"  - {col}: {cnt} 個")
# else:
#     print("所有數值欄位都沒有 inf 或 -inf。")

# # 2. 進一步查看這些欄位在哪些列(row)有無限值
# for col in inf_counts:
#     # 找出該欄位是 inf 或 -inf 的列索引
#     inf_rows = df.index[np.isinf(df[col])]
#     print(f"\n欄位 '{col}' 在以下列有無限值：")
#     for r in inf_rows:
#         print(f"  - 列索引: {r}, 值: {df.at[r, col]}")

# --------------------------------------------------

# import pandas as pd
# import numpy as np

# # 讀取資料
# df = pd.read_csv("./UNSW-NB15/UNSW-NB15-feature-clean.csv")

# print("清理前資料形狀:", df.shape)

# # 過濾掉在 'SRC_TO_DST_SECOND_BYTES' 或 'DST_TO_SRC_SECOND_BYTES' 欄位中有無限值的列
# mask = ~(np.isinf(df['SRC_TO_DST_SECOND_BYTES']) | np.isinf(df['DST_TO_SRC_SECOND_BYTES']))
# df_clean = df[mask].copy()

# print("清理後資料形狀:", df_clean.shape)

# # 輸出處理後的資料到新檔案
# output_file = "./UNSW-NB15/UNSW-NB15-feature-clean.csv"
# df_clean.to_csv(output_file, index=False)
# print(f"已儲存清理後的資料至 {output_file}")

#----------------------------------------
# # 做 LOF
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import LocalOutlierFactor

# # 1. 讀入資料
# df = pd.read_csv("./ctu13 contrastive/ctu13_clean.csv")  # 假設已經含有 'Label' 欄
# print("原始各類別數量：")
# print(df["Label"].value_counts())

# # 2. LOF + 5% contamination 清洗每個類別
# cleaned_parts = []
# features = df.columns.drop("Label")

# for lab, group in df.groupby("Label"):
#     X = group[features].values
#     # 先做標準化
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     # LOF 檢測
#     lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
#     pred = lof.fit_predict(X_scaled)   # inlier: 1, outlier: -1
#     grp_clean = group[pred == 1].copy()
#     print(f"Label={lab}：原 {len(group)} → 清洗後 {len(grp_clean)}")
#     cleaned_parts.append(grp_clean)

# df_clean = pd.concat(cleaned_parts, ignore_index=True)
# print("\nLOF 清洗後總量：")
# print(df_clean["Label"].value_counts())

# # 3. 依需求對各類別進行上下採樣
# resampled = []

# def resample_group(data, target_size):
#     current = len(data)
#     if current == target_size:
#         return data
#     elif current > target_size:
#         return data.sample(n=target_size, random_state=42)
#     else:
#         return data.sample(n=target_size, replace=True, random_state=42)

# # 定義每個 Label 最終要的數量
# target_counts = {
#     0: 160000,   # 類別 0 放大到 110k
#     1:   5000,   # 類別 1 縮小到 5k
#     3:   2000,   # 類別 3 縮小到 2k
#     5:   8000,   # 類別 5 縮小到 8k
#     6:   1000    # 類別 6 縮小到 1k
#     # 類別 2、4 未列出 → 保持 LOF 清洗後的原始數量
# }

# for lab, group in df_clean.groupby("Label"):
#     if lab in target_counts:
#         grp_res = resample_group(group, target_counts[lab])
#         print(f"Label={lab}：從 {len(group)} → 重採樣到 {len(grp_res)}")
#     else:
#         grp_res = group
#         print(f"Label={lab}：保持 {len(group)}")
#     resampled.append(grp_res)

# df_final = pd.concat(resampled, ignore_index=True)
# print("\n最終各類別分布：")
# print(df_final["Label"].value_counts())

# # 4. 儲存結果
# df_final.to_csv("./ctu13 contrastive/ctu13_cleaned_resampled.csv", index=False)
# print("\n已將清洗並重採樣後的資料存成：")
# print("  - ctu13_cleaned_resampled.csv")

#---------------------------------------
import pandas as pd

file_path = './ctu13 contrastive/ctu13_cleaned_resampled.csv'
df = pd.read_csv(file_path)

print("特徵名稱：", list(df.columns))
print("前十行內容:\n", df.head(10))

print("資料形狀:", df.shape)
# 顯示每個檔案的 label 分布情況
label_counts = df['Label'].value_counts()
print(f"Label 分布:")
print(label_counts)

# # # === 將 DataFrame 輸出為 CSV 檔 ===
# # output_csv = 'ctu13_label.csv'
# # df.to_csv(output_csv, index=False)
# # print(f"已將 DataFrame 存成 CSV：{output_csv}")