def remove_smiles_lines(input_file, output_file):
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            # 过滤包含 "smiles" 的行（不区分大小写）
            if "smiles" not in line.lower():
                line = line.strip()
                if not line:
                    continue  # 跳过空行

                # 按空格分割，取第一个元素（SMILES）
                smiles = line.split()[0]
                f_out.write(smiles + "\n")

# 使用示例
remove_smiles_lines("zinc.smi", "zinc_clean.smi")