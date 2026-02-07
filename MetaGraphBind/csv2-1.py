import pandas as pd

# 读取文件一和文件二
file1 = pd.read_csv('data/gdb_clean.csv')  # 请根据文件路径修改
file2 = pd.read_csv('6.9_val_out.csv')  # 请根据文件路径修改

# 提取文件一的七列
file1_selected = file1[['ID', 'SMILES', 'Metal', 'Medium', 'Solvent', 't', 'I-str']]

# 提取文件二的Pre列
file2_selected = file2[['Pre']]

# 创建空的列表来存储最终的行
final_rows = []

# 定义最终列名
columns = ['ID', 'SMILES', 'Metal', 'Medium', 'Solvent', 't', 'I-str', 'Am', 'Eu', 'det']

# 遍历每两行
for i in range(0, len(file1), 2):
    # 上面一行数据不变
    row1 = file1_selected.iloc[i]

    # 下面一行的Pre列作为Am，并计算det
    row2 = file2_selected.iloc[i + 1]

    # 获取Am和Eu的值
    Am = row2['Pre']
    Eu = file2_selected.iloc[i]['Pre']  # 将Pre改为Eu

    # 创建det列，Am - Eu
    det = Am - Eu

    # 使用pd.concat来合并数据，并指定列名
    final_row = pd.concat([row1, pd.Series({'Am': Am, 'Eu': Eu, 'det': det})], ignore_index=True)

    # 添加到final_rows列表
    final_rows.append(final_row)

# 将最终的结果转换为DataFrame，并指定列名
final_df = pd.DataFrame(final_rows, columns=columns)


# 保存到新的Excel文件
final_df.to_excel('Am-Eu_6.19.xlsx', index=False)