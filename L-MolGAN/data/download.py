def remove_smiles_lines(input_file, output_file):
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            # Filter the rows that contain "smiles" (case-insensitive)
            if "smiles" not in line.lower():
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                # Split by spaces and take the first element (SMILES)
                smiles = line.split()[0]
                f_out.write(smiles + "\n")

# 使用示例
remove_smiles_lines("zinc.smi", "zinc_clean.smi")