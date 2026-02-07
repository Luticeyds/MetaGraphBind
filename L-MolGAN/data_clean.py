import re
from rdkit import Chem
from rdkit.Chem import BondType, SanitizeFlags


def read_smi_file(file_path):
    """ 读取 .smi 文件并返回 SMILES 列表 """
    smiles_list = []
    with open(file_path, 'r') as file:
        for line in file:
            smiles = line.strip().split()[0]  # 只取 SMILES，忽略可能存在的分子名称
            if smiles:
                smiles_list.append(smiles)
    return smiles_list


def filter_molecules(smiles_list):
    """过滤分子并修改键类型：
    1. 转换小写字母c,n,s,o为大写（去除芳香性）
    2. 处理双、三键（保留O=，=O，C#N）
    """
    filtered_smiles = []

    for smiles in smiles_list:
        # 1
        s = ''.join([c.upper() if c in {'c', 'n', 's', 'o'} else c for c in smiles])
        # 2
        s = s.replace('O=', 'TEMP_O_EQUAL').replace('=O', 'TEMP_EQUAL_O').replace('C#N', 'TEMP_C_N').replace('N#C', 'TEMP_N_C')
        s = s.replace('=', '').replace('#', '')
        s = s.replace('TEMP_O_EQUAL', 'O=').replace('TEMP_EQUAL_O', '=O').replace('TEMP_C_N', 'C#N').replace('TEMP_N_C', 'N#C')

        # 读取分子并跳过无效SMILES
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        if mol.GetNumAtoms() > 9:
            continue

        new_smiles = Chem.MolToSmiles(mol)
        if new_smiles not in filtered_smiles:
            filtered_smiles.append(new_smiles)

    return filtered_smiles


def write_smi_file(output_path, smiles_list):
    """ 将筛选后的 SMILES 写入文件 """
    with open(output_path, 'w') as file:
        for smiles in smiles_list:
            file.write(smiles + '\n')

# 运行筛选
input_smi = 'gdb/gdb.txt'  # 输入文件名
output_smi = 'gdb/gdb_clean.smi'  # 输出文件名

# 读取 .smi 文件
smiles_list = read_smi_file(input_smi)

# 筛选分子
filtered_smiles = filter_molecules(smiles_list)

# 保存筛选后的分子
write_smi_file(output_smi, filtered_smiles)

print(f"筛选完成，结果保存在 {output_smi}")

# # 读取SDF文件
# suppl = Chem.SDMolSupplier('data/gdb9.sdf')
# t = 0
# fa = 0
# with open('data/gdb9.smi', 'w') as f:
#     for mol in suppl:
#         if mol is not None:
#             smiles = Chem.MolToSmiles(mol)
#             # 获取分子名称（假设名称存储在'_Name'属性中）
#             f.write(f"{smiles}\n")
#             t += 1
#         else:
#             print("Warning: 跳过一个无法解析的分子")
#             fa += 1
# print(f"T:{t},F{fa}")