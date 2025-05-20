import pandas as pd
import plotly.express as px
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 1. 讀取資料
file_path = './ctu13 contrastive/ctu13_density_simple_embed.csv'
df = pd.read_csv(file_path)

print("資料形狀:", df.shape)

label_counts = df['Label'].value_counts()
print(f"Label 分布:")
print(label_counts)

# 2. 分割特徵與標籤
X = df.drop(['Label'], axis=1)
y = df['Label']

# 3. 資料標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 使用 t-SNE 將資料降維到3D
# 注意：t-SNE 對於資料量大時計算量較高，可以酌情調整 perplexity、n_iter 等參數
tsne = TSNE(n_components=3, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 5. 將 t-SNE 結果與 Label 合併為一個 DataFrame，方便繪圖
df_plot = pd.DataFrame({
    'X': X_tsne[:, 0],
    'Y': X_tsne[:, 1],
    'Z': X_tsne[:, 2],
    'Label': y
})

# 6. 使用 Plotly Express 進行互動式 3D 視覺化
fig = px.scatter_3d(
    df_plot, 
    x='X', 
    y='Y', 
    z='Z', 
    color='Label',
    # title='CTU13 Original'
    # title='CTU13 Embeddings'
    title= 'CTU13 density'
)
fig.show()
