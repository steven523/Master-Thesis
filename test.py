# import pandas as pd
# import numpy as np

# # 讀取 ctu13_label.pkl 檔案
# pkl_file_path = './df9/df9_all_100000.pkl'
# try:
#     pkl_data = pd.read_pickle(pkl_file_path)
#     print("\n=== df9_allFeature.pkl ===")
#     print("資料形狀:", pkl_data.shape)
#     # print("前五行內容:\n", pkl_data.head())
#     # 查看前 20 行內容
#     print("前二十行內容:\n", pkl_data.head(20))
#     print("\n每個特徵的名稱:")
#     print(pkl_data.columns)
# except Exception as e:
#     print(f"無法讀取 {pkl_file_path}: {e}")

# # 讀取 pkl 檔案
# ctu13_data = pd.read_pickle(pkl_file_path)

# # 檢查有哪些 label 和每個 label 的資料數量
# label_counts = ctu13_data['label'].value_counts()

# print(f"特徵名稱：{list(ctu13_data.columns)}")

# # 顯示結果
# print("Label 分布:")
# print(label_counts)

# -----------------------------------------------------

import pandas as pd
import os

# 設定資料夾路徑
folder_path = './CSECICIDS2018_improved'  # 根據你的資料夾路徑做調整
file_names = ['Friday-02-03-2018', 'Friday-16-02-2018', 'Friday-23-02-2018', 'Thursday-01-03-2018', 'Thursday-15-02-2018',
              'Thursday-22-02-2018', 'Tuesday-20-02-2018', 'Wednesday-14-02-2018', 'Wednesday-21-02-2018', 'Wednesday-28-02-2018']

# 創建一個空的 DataFrame 用來儲存合併後的資料
all_data = pd.DataFrame()

# data = pd.read_csv(folder_path)
# print(f"\n=== {folder_path}.csv ===")
# print("資料形狀:", data.shape)
# print("前二十行內容:\n", data.head(20))

# # 顯示每個檔案的 label 分布情況
# label_counts = data['Label'].value_counts()
# print(f"{folder_path} 的 Label 分布:")
# print(label_counts)

# 讀取每一個檔案並合併
for file_name in file_names:
    file_path = os.path.join(folder_path, f'{file_name}.csv')
    try:
        # 讀取 CSV 檔案
        data = pd.read_csv(file_path)
        # 顯示檔案的前20行內容
        print(f"\n=== {file_name}.csv ===")
        print("資料形狀:", data.shape)
        print("前二十行內容:\n", data.head(20))
        
        # 顯示每個檔案的 label 分布情況
        label_counts = data['Label'].value_counts()
        print(f"{file_name} 的 Label 分布:")
        print(label_counts)
        
    #     # 合併資料
    #     all_data = pd.concat([all_data, data], axis=0, ignore_index=True)
    except Exception as e:
        print(f"無法讀取 {file_path}: {e}")

# 顯示所有資料的特徵名稱
print("\n資料合併完成。")
print("特徵名稱：", list(all_data.columns))
print("資料形狀:", all_data.shape)

# 檢查資料中的 label 分布情況
label_counts_all = all_data['Label'].value_counts()
print("合併後的 Label 分布:")
print(label_counts_all)

# 將合併後的資料輸出成新的 CSV 檔案
output_file = os.path.join(folder_path, 'merged_CICIDS2017.csv')
all_data.to_csv(output_file, index=False)
print(f"\n合併後的資料已輸出至: {output_file}")