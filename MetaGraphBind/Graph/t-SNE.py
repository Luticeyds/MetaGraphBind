import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm


def smiles_to_fps(smiles_list):
    fps = []
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fps.append(list(fp))
            valid_smiles.append(smi)
    return np.array(fps), valid_smiles

# 读取 Excel 所有 sheet
file_path = 'smiles.xlsx'  # 替换为你的文件路径
sheet_A = pd.read_excel(file_path, sheet_name='Jess')
sheet_B = pd.read_excel(file_path, sheet_name='mid')
sheet_C = pd.read_excel(file_path, sheet_name='bro')
sheet_D = pd.read_excel(file_path, sheet_name='nola')
sheet_E = pd.read_excel(file_path, sheet_name='new_no')

# 指定 SMILES 列名（请确认列名是否为 "SMILES"）
name_A = 'Jess'
name_B = 'mid'
name_C = 'Article'
name_D = 'no_label'
name_E = 'no_label_new'

all_smiles_dict = {
    name_A: sheet_A['SMILES'].dropna().tolist(),
    name_B: sheet_B['SMILES'].dropna().tolist(),
    name_C: sheet_C['SMILES'].dropna().tolist(),
    name_D: sheet_D['SMILES'].dropna().tolist(),
    name_E: sheet_E['SMILES'].dropna().tolist()
}

data_work = [name_C, name_D, name_E]
all_fps = []
all_labels = []
for name in data_work:
    fps, _ = smiles_to_fps(all_smiles_dict[name])
    all_fps.append(fps)
    all_labels.extend([name] * len(fps))



# 转为数组
fps_all = np.vstack(all_fps)

# 设置合理的 perplexity

# t-SNE 降维
tsne = TSNE(n_components=2, perplexity=100, random_state=42, n_iter=1000)
coords = tsne.fit_transform(fps_all)

# 可视化（自动分配颜色）
unique_labels = sorted(set(all_labels))
color_map = {name_A: '#fed71a',  # 佛手黄
             name_B: '#4daf4a',  # 绿色 Green
             name_C: '#152732',  # 黑
             name_D: '#e41a1c',  # 红色 Red
             name_E: '#87C0CA'}  # 蓝色 Blue
plt.figure(figsize=(8, 6))
for i, label in enumerate(unique_labels):
    idx = [j for j, l in enumerate(all_labels) if l == label]
    plt.scatter(coords[idx, 0], coords[idx, 1],
                color=color_map[label],
                label=label,
                alpha=0.5,
                s=5)                      # 点大小

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SEN")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig('CDE-100.png', dpi=1000, bbox_inches="tight")
plt.close()