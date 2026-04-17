import re
from rdkit import Chem
from rdkit.Chem import BondType, SanitizeFlags


def read_smi_file(file_path):
    """ Read .smi file and return a list of SMILES """
    smiles_list = []
    with open(file_path, 'r') as file:
        for line in file:
            smiles = line.strip().split()[0]  # Only extract SMILES and disregard any potential molecular names
            if smiles:
                smiles_list.append(smiles)
    return smiles_list


def filter_molecules(smiles_list):
    """
    Filter molecules and modify bond types:

    1. Convert lowercase letters c, n, s, o to uppercase (remove aromaticity)
    2. Handle double and triple bonds (preserve O=, =O, C#N)
    """
    filtered_smiles = []

    for smiles in smiles_list:
        # 1
        s = ''.join([c.upper() if c in {'c', 'n', 's', 'o'} else c for c in smiles])
        # 2
        s = s.replace('O=', 'TEMP_O_EQUAL').replace('=O', 'TEMP_EQUAL_O').replace('C#N', 'TEMP_C_N').replace('N#C', 'TEMP_N_C')
        s = s.replace('=', '').replace('#', '')
        s = s.replace('TEMP_O_EQUAL', 'O=').replace('TEMP_EQUAL_O', '=O').replace('TEMP_C_N', 'C#N').replace('TEMP_N_C', 'N#C')

        # Read molecules and skip invalid SMILES
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
    """ Write the filtered SMILES into a file """
    with open(output_path, 'w') as file:
        for smiles in smiles_list:
            file.write(smiles + '\n')

# Run filter
input_smi = 'gdb/gdb.txt'  # input file name
output_smi = 'gdb/gdb_clean.smi'  # output filename

# Read .smi files
smiles_list = read_smi_file(input_smi)

# screening molecules
filtered_smiles = filter_molecules(smiles_list)

# Save the filtered molecules
write_smi_file(output_smi, filtered_smiles)

print(f"The filtering process has been completed, and the results are saved in {output_smi}")

