from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import pandas as pd


def add_rf_to_c_atoms(mol):
    unique_mols = set()
    result_mols = []

    num_atoms = mol.GetNumAtoms()

    # 遍历所有原子
    for atom_idx in range(num_atoms):
        editable_mol = Chem.RWMol(mol)
        atom = mol.GetAtomWithIdx(atom_idx)
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) != 1:
            continue

        if atom.GetAtomicNum() not in [6, 7, 8, 15]:  # C N O P
            continue

        if atom.IsInRing():
            continue

        num_hydrogens = atom.GetTotalNumHs()
        if num_hydrogens == 0:
            continue

        rf_atom = Chem.Atom(104)  # 'Rf'
        rf_idx = editable_mol.AddAtom(rf_atom)

        editable_mol.AddBond(atom_idx, rf_idx, Chem.BondType.SINGLE)

        new_mol = editable_mol.GetMol()
        new_smiles = Chem.MolToSmiles(new_mol)

        if new_smiles not in unique_mols:
            unique_mols.add(new_smiles)
            result_mols.append(new_mol)

    double_mols = add_rf_mt_to_mol(mol)

    return result_mols, double_mols

def add_rf_mt_to_mol(mol):
    """查找所有不在环中的五碳链，并在首尾添加标识符'Cf'、'Es'。"""
    paths = []
    modified_mols = []
    for atom in mol.GetAtoms():
        # 获取从该原子出发的所有长度为4的路径（5个原子）
        ps = Chem.FindAllPathsOfLengthN(mol, 4, rootedAtAtom=atom.GetIdx(), useBonds=False)
        paths.extend(ps)

    # 过滤条件：所有原子为碳且不在环中
    carbon_paths = []
    for path in paths:
        all_carbon = True
        in_ring = False
        for atom_idx in path:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetSymbol() != 'C':
                all_carbon = False
                break
            if atom.IsInRing():
                in_ring = True
                break
        if all_carbon and not in_ring:
            carbon_paths.append(path)

    # 去重处理
    seen = set()
    for path in carbon_paths:
        # 标准化路径表示（小端在前）
        if path[0] > path[-1]:
            path = tuple(reversed(path))
        key = (path[0], path[-1]) + tuple(sorted(path[1:-1]))
        if key not in seen:
            seen.add(key)

            try:
                ed_mol = Chem.EditableMol(Chem.Mol(mol))
                # 在路径首尾添加Cf
                start_idx = path[0]
                end_idx = path[-1]
                if mol.GetAtomWithIdx(start_idx).GetTotalNumHs() == 0 or mol.GetAtomWithIdx(end_idx).GetTotalNumHs() == 0:
                    continue
                new_c1 = ed_mol.AddAtom(Chem.Atom(104))
                new_c2 = ed_mol.AddAtom(Chem.Atom(109))
                ed_mol.AddBond(start_idx, new_c1, Chem.BondType.SINGLE)
                ed_mol.AddBond(end_idx, new_c2, Chem.BondType.SINGLE)

                # 生成新分子并验证
                new_mol = ed_mol.GetMol()
                Chem.SanitizeMol(new_mol)
                modified_mols.append(new_mol)


            except:
                continue

    return modified_mols


def get_neiid_bysymbol(combo, symbol):
    for at in combo.GetAtoms():
        if at.GetSymbol() == symbol:
            neighbors = at.GetNeighbors()
            if neighbors:
                at_nei = neighbors[0]
                return at_nei.GetIdx()

def get_id_bysymbol(combo, symbol):
    for at in combo.GetAtoms():
        if at.GetSymbol() == symbol:
            return at.GetIdx()

def combine_leaf_and_core(core_mol, leaf_mol, L, A):
    # Combine molecular core and a molecular leaf in one unit
    combo = Chem.CombineMols(leaf_mol, core_mol)
    back = combine_oneself(combo, L, A)

    return back

def combine_oneself(combo, L, A):
    # Combine molecular core and a molecular leaf in one unit

    A_NEI_ID = get_neiid_bysymbol(combo, A)
    L_NEI_ID = get_neiid_bysymbol(combo, L)

    edcombo = Chem.EditableMol(combo)
    edcombo.AddBond(A_NEI_ID, L_NEI_ID, order=Chem.BondType.SINGLE)

    L_ID = get_id_bysymbol(combo, L)

    edcombo.RemoveAtom(L_ID)
    back = edcombo.GetMol()

    A_ID = get_id_bysymbol(back, A)

    edcombo.RemoveAtom(A_ID)
    back = edcombo.GetMol()

    return back

def delete_id(mol, A):
    edcombo = Chem.EditableMol(mol)
    A_ID = get_id_bysymbol(mol, A)
    edcombo.RemoveAtom(A_ID)
    back = edcombo.GetMol()
    return back

def mol_combine(single_mols, double_mols, core_mols_file='data/core_mols.4.16.pkl'):
    result_smi = []
    L = 'Rf'
    L2 = 'Mt'
    A1 = 'Db'
    A2 = 'Sg'
    A3 = 'Bh'
    A4 = 'Hs'
    B1 = 'Nh'
    B2 = 'Ds'
    B3 = 'Rg'
    B4 = 'Cn'
    with open(core_mols_file, 'rb') as file:
        core_mols = pickle.load(file)

    for core_mol in core_mols:
        # 针对每个核心分子执行替代
        for leaf_mol in single_mols:
            # 检查并替代 A1 位点
            if any(atom.GetSymbol() == A1 for atom in core_mol.GetAtoms()):
                mid_mol1 = combine_leaf_and_core(core_mol, leaf_mol, L, A1)
                # 检查并替代 A2 位点
                if any(atom.GetSymbol() == A2 for atom in mid_mol1.GetAtoms()):
                    mid_mol2 = combine_leaf_and_core(mid_mol1, leaf_mol, L, A2)
                else:
                    mid_mol2 = mid_mol1

                # 检查并替代 A3 位点
                if any(atom.GetSymbol() == A3 for atom in mid_mol2.GetAtoms()):
                    mid_mol3 = combine_leaf_and_core(mid_mol2, leaf_mol, L, A3)
                else:
                    mid_mol3 = mid_mol2

                # 检查并替代 A4 位点
                if any(atom.GetSymbol() == A4 for atom in mid_mol3.GetAtoms()):
                    mid_mol4 = combine_leaf_and_core(mid_mol3, leaf_mol, L, A4)
                else:
                    mid_mol4 = mid_mol3
                # 转换为 SMILES 表示
                smi = Chem.MolToSmiles(mid_mol4)
                result_smi.append(smi)


        if len(double_mols) > 0:
            for leaf_mol in double_mols:
                # 检查并替代 B1/B2 位点
                if (any(atom.GetSymbol() == B1 for atom in core_mol.GetAtoms()) and
                        any(atom1.GetSymbol() == B2 for atom1 in core_mol.GetAtoms())):
                    mid_mol1 = combine_leaf_and_core(core_mol, leaf_mol, L, B1)
                    mid_mol2 = combine_oneself(mid_mol1, L2, B2)

                    if (any(atom.GetSymbol() == B3 for atom in mid_mol2.GetAtoms()) and
                            any(atom2.GetSymbol() == B4 for atom2 in mid_mol2.GetAtoms())):
                        mid_mol3 = combine_leaf_and_core(mid_mol2, leaf_mol, L, B3)
                        mid_mol4 = combine_oneself(mid_mol3, L2, B4)
                    else:
                        mid_mol4 = mid_mol2
                    smi = Chem.MolToSmiles(mid_mol4)
                    result_smi.append(smi)

    return result_smi

def read_smi_file(file_path):
    """ 读取 .smi 文件并返回 SMILES 列表 """
    smiles_list = []
    with open(file_path, 'r') as file:
        for line in file:
            smiles = line.strip().split()[0]  # 只取 SMILES，忽略可能存在的分子名称
            if smiles:
                smiles_list.append(smiles)
    return smiles_list

def main(txt_file, out_file='zinc/data_4.10.xlsx', smiles_file='gdb/gdb_clean_smiles.csv'):
    leaf_list = read_smi_file(txt_file)
    feature_data = pd.read_pickle('data/feature.pkl')
    resule_smiles = []
    id_list = []
    id = 0
    for smi in leaf_list:
        mol = Chem.MolFromSmiles(smi)
        single, double = add_rf_to_c_atoms(mol)
        re_smi = mol_combine(single, double)
        if len(re_smi) != 0:
            id += 1
            resule_smiles.extend(re_smi)
            id_list.extend([id]*len(re_smi))

    # 保存 resule_smiles 到一个单独的数据表
    smiles_df = pd.DataFrame({'ID': id_list, 'SMILES': resule_smiles})
    smiles_df.to_csv(smiles_file, index=False)

    # result_data = []
    # feature_data['W'] = feature_data['W'].astype(int)
    # for smi, mol_id in zip(resule_smiles, id_list):
    #     rows_to_add = feature_data[feature_data['W'] != ''].copy()
    #     for index, row in rows_to_add.iterrows():
    #         row['SMILES'] = smi
    #         row['ID'] = mol_id
    #         result_data.append(row)
    #
    #
    # result_df = pd.DataFrame(result_data)
    # result_df.to_csv(out_file, index=False)



if __name__ == '__main__':
    main('gdb/gdb_clean.smi', 'gdb/gdb_clean.csv')