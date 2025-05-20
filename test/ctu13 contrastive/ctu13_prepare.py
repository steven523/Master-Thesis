import pandas as pd
import numpy as np

# 讀取 ctu13_label.pkl 檔案
pkl_file_path = 'ctu13_label.pkl'
try:
    pkl_data = pd.read_pickle(pkl_file_path)
    print("\n=== ctu13_label.pkl ===")
    print("資料形狀:", pkl_data.shape)
    # print("前五行內容:\n", pkl_data.head())
    # 查看前 20 行內容
    print("前二十行內容:\n", pkl_data.head(20))
    print("\n每個特徵的名稱:")
    print(pkl_data.columns)
except Exception as e:
    print(f"無法讀取 {pkl_file_path}: {e}")

# 讀取 ctu13_rl_label.npy 檔案
npy_file_path = 'ctu13_rl_label.npy'
try:
    npy_data = np.load(npy_file_path)
    print("\n=== ctu13_rl_label.npy ===")
    print("資料形狀:", npy_data.shape)
    print("前十個元素:", npy_data[:20])
except Exception as e:
    print(f"無法讀取 {npy_file_path}: {e}")

# 讀取 pkl 檔案
ctu13_data = pd.read_pickle(pkl_file_path)

# 檢查有哪些 label 和每個 label 的資料數量
label_counts = ctu13_data['label'].value_counts()

print(f"特徵名稱：{list(ctu13_data.columns)}")

# 顯示結果
print("Label 分布:")
print(label_counts)

#-----------------------------------------------
# 以下是對 ctu13 各類別資料量做擴增，利用最近鄰演算法來生成新樣本

# import pandas as pd
# import numpy as np
# from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import StandardScaler

# # 讀取 ctu13 資料
# pkl_path = 'ctu13_label.pkl'
# npy_path = 'ctu13_rl_label.npy'
# ctu13_data = pd.read_pickle(pkl_path)
# ctu13_labels = np.load(npy_path)

# # 將 DataFrame 中加入 label 欄位
# ctu13_data['label'] = ctu13_labels

# # 選擇數值型特徵
# numeric_features = ctu13_data.select_dtypes(include=[np.number])

# # 標準化數據
# scaler = StandardScaler()
# numeric_features_scaled = scaler.fit_transform(numeric_features)

# # 設定擴展資料量至 100 萬筆
# required_data_len = 500000
# current_data_len = len(ctu13_data)
# additional_samples_needed = required_data_len - current_data_len

# print(f"Current data length: {current_data_len}")
# print(f"Generating {additional_samples_needed} additional samples...")

# # 使用最近鄰來生成新樣本
# nbrs = NearestNeighbors(n_neighbors=5).fit(numeric_features_scaled)
# _, indices = nbrs.kneighbors(numeric_features_scaled)

# new_samples = []
# new_labels = []

# for _ in range(additional_samples_needed):
#     idx = np.random.randint(0, current_data_len)
#     neighbors = indices[idx]
#     chosen_idx = np.random.choice(neighbors)
#     new_sample = numeric_features_scaled[idx] + np.random.rand() * (numeric_features_scaled[chosen_idx] - numeric_features_scaled[idx])
#     new_samples.append(new_sample)
#     new_labels.append(ctu13_labels[idx])

# # 將新生成的樣本和標籤加入原始資料
# new_samples = scaler.inverse_transform(new_samples)  # 將數據轉換回原始尺度
# new_samples_df = pd.DataFrame(new_samples, columns=numeric_features.columns)
# new_samples_df['label'] = new_labels

# # 合併原始資料和新生成的樣本
# augmented_data = pd.concat([ctu13_data, new_samples_df], axis=0).reset_index(drop=True)

# # 檢查最終資料分布
# print("最終 Label 分布:")
# print(augmented_data['label'].value_counts())

# # 儲存擴展後的資料
# augmented_data.to_pickle('./ctu13_label_augmented.pkl')
# np.save('./ctu13_rl_label_augmented.npy', augmented_data['label'].values)

# print("資料擴展完成")

#-----------------------------------------------
# # 以下是對 ctu13 各類別資料量做平衡

import pandas as pd

# 读取数据
file_path = 'ctu13_label.pkl'
netflow_data = pd.read_pickle(file_path)

# 查看标签的分布
label_distribution = netflow_data['label'].value_counts()
print("原始标签分布:")
print(label_distribution)

# 设置目标总数据量和类别均衡后的每个类别的数据量
target_total = 100000
num_classes = len(label_distribution)
target_per_class = target_total // num_classes

# 重新采样每个类别的数据
balanced_data = []
for label in label_distribution.index:
    class_data = netflow_data[netflow_data['label'] == label]
    balanced_class_data = class_data.sample(n=target_per_class, random_state=42, replace=len(class_data) < target_per_class)
    balanced_data.append(balanced_class_data)

# 合并所有类别的数据
balanced_data = pd.concat(balanced_data).sample(frac=1, random_state=42).reset_index(drop=True)

# 查看新的数据集分布
print("\n新标签分布:")
print(balanced_data['label'].value_counts())
print("資料形狀:", balanced_data.shape)

# 保存新的数据集
balanced_file_path = 'ctu13_label_100000.pkl'
balanced_data.to_pickle(balanced_file_path)
print(f"\n平衡后的数据集已保存到: {balanced_file_path}")


