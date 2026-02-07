import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import numpy as np
from tqdm import tqdm  # 导入 tqdm 库以显示进度条

# 输入.xlsx 文件路径
file_path = "unlabeled.xlsx"

# 读取 Excel 文件中的 SMILES 列（前 10%）
df = pd.read_excel(file_path)
smiles_list = df["SMILES"].head(int(len(df) * 0.1))

# -------------------------
# 计算分子指纹
# -------------------------
def calculate_fingerprints(smiles_list):
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
           for mol in tqdm(mols, desc="计算分子指纹")]
    return np.array(fps)

# -------------------------
# Tanimoto 核函数
# -------------------------
def tanimoto_kernel(X):
    return 1 - pdist(X, metric="jaccard")

# -------------------------
# 初始化 t-SNE
# -------------------------
tsne = TSNE(
    n_components=2,
    metric="precomputed",
    random_state=42,
    init='random',
    learning_rate='auto'
)

# -------- 执行计算 --------
print("计算分子指纹...")
fingerprints = calculate_fingerprints(smiles_list)

print("计算 Tanimoto 距离矩阵...")
tanimoto_dist = squareform(tanimoto_kernel(fingerprints))

print("执行 t-SNE 降维...")
tsne_results = tsne.fit_transform(1 - tanimoto_dist)

# -------- 绘图 --------
plt.figure(figsize=(10, 8))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], color="#845ec2", s=10)

plt.title("t-SNE Visualization (Top 10% SMILES from XLSX)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

# 保存图像
output_path = "unlabeled_t-sen.svg"
plt.savefig(output_path, format='svg')

print(f"图像已保存至 {output_path}")