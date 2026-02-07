from rdkit import Chem
import pandas as pd
import pickle
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


    A_NEI_ID = get_neiid_bysymbol(combo, A)
    L_NEI_ID = get_neiid_bysymbol(combo, L)


    edcombo = Chem.EditableMol(combo)
    edcombo.AddBond(A_NEI_ID, L_NEI_ID, order=Chem.rdchem.BondType.SINGLE)

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

def mol_combine(excel_file='mol_link.xlsx', output_file_path='mol_combine.12.3.xlsx', mode='对称'):
    core_data = pd.read_excel(excel_file, sheet_name='Sheet1')

    leaf_data = pd.read_excel(excel_file, sheet_name='Sheet2')
    feature_data = pd.read_excel(excel_file, sheet_name='Sheet5')
    feature_data.to_pickle('feature.pkl')
    print('all')
    core_smiles = core_data['smiles'].tolist()
    core_sub_points = core_data['取代位点数'].tolist()
    core_type = core_data['配体种类'].tolist()
    core_id = core_data['id'].tolist()
    leaf_smiles = leaf_data['smiles'].tolist()
    leaf_type = leaf_data['种类'].tolist()
    core_smiles = [smiles for smiles in core_smiles if isinstance(smiles, str)]
    core_mols = [Chem.MolFromSmiles(smiles) for smiles in core_smiles]
    with open('core_mols.pkl', 'wb') as f:
        pickle.dump(core_mols, f)
    leaf_mols = [Chem.MolFromSmiles(smiles) for smiles in leaf_smiles]
    zero_mol = Chem.MolFromSmiles('C')
    leaf_mols.append(zero_mol)
    result_smi = []
    result_types = []
    result_leaves1 = []
    result_leaves2 = []
    result_leaves3 = []
    result_leaves4 = []
    result_ids = []
    L = 'Rf'
    A1 = 'Db'
    A2 = 'Sg'
    A3 = 'Bh'
    A4 = 'Hs'
    if mode == '非对称':
        for i, core_mol in enumerate(core_mols):
            point = int(core_sub_points[i])
            type_id = core_type[i]
            id = core_id[i]
            if point == 1:
                for j, leaf_mol in enumerate(leaf_mols):
                    if leaf_mol == zero_mol:
                        leaf_id1 = leaf_type[0]
                        mid_mol = delete_id(core_mol, A1)
                    else:
                        leaf_id1 = leaf_type[j]
                        mid_mol = combine_leaf_and_core(core_mol, leaf_mol, L, A1)
                    smi = Chem.MolToSmiles(mid_mol)
                    if smi not in result_smi:
                        result_smi.append(smi)
                        result_types.append(type_id)
                        result_leaves1.append(leaf_id1)
                        result_leaves2.append(0)
                        result_leaves3.append(0)
                        result_leaves4.append(0)
                        result_ids.append(id)

            if point == 2:
                for j1, leaf_mol1 in enumerate(leaf_mols):
                    if leaf_mol1 == zero_mol:
                        leaf_id1 = 0
                        mid_mol1 = delete_id(core_mol, A1)
                    else:
                        leaf_id1 = leaf_type[j1]
                        mid_mol1 = combine_leaf_and_core(core_mol, leaf_mol1, L, A1)
                    for j2, leaf_mol2 in enumerate(leaf_mols):
                        if leaf_mol2 == zero_mol:
                            leaf_id2 = 0
                            mid_mol2 = delete_id(mid_mol1, A2)
                        else:
                            leaf_id2 = leaf_type[j2]
                            mid_mol2 = combine_leaf_and_core(mid_mol1, leaf_mol2, L, A2)
                        smi = Chem.MolToSmiles(mid_mol2)
                        if smi not in result_smi:
                            result_smi.append(smi)
                            result_types.append(type_id)
                            result_leaves1.append(leaf_id1)
                            result_leaves2.append(leaf_id2)
                            result_leaves3.append(0)
                            result_leaves4.append(0)
                            result_ids.append(id)

            if point == 3:
                for j, leaf_mol in enumerate(leaf_mols):
                    if leaf_mol == zero_mol:
                        leaf_id1 = 0
                        leaf_id2 = 0
                        leaf_id3 = 0
                        mid_mol = delete_id(core_mol, A1)
                        mid_mol = delete_id(mid_mol, A2)
                        mid_mol = delete_id(mid_mol, A3)
                    else:
                        leaf_id1 = leaf_type[j]
                        leaf_id2 = leaf_type[j]
                        leaf_id3 = leaf_type[j]
                        mid_mol = combine_leaf_and_core(core_mol, leaf_mol, L, A1)
                        mid_mol = combine_leaf_and_core(mid_mol, leaf_mol, L, A2)
                        mid_mol = combine_leaf_and_core(mid_mol, leaf_mol, L, A3)
                    smi = Chem.MolToSmiles(mid_mol)
                    if smi not in result_smi:
                        result_smi.append(smi)
                        result_types.append(type_id)
                        result_leaves1.append(leaf_id1)
                        result_leaves2.append(leaf_id2)
                        result_leaves3.append(leaf_id3)
                        result_leaves4.append(0)
                        result_ids.append(id)
            if point == 4:
                for j1, leaf_mol1 in enumerate(leaf_mols):
                    if leaf_mol1 == zero_mol:
                        leaf_id1 = 0
                        leaf_id2 = 0
                        mid_mol1 = delete_id(core_mol, A1)
                        mid_mol1 = delete_id(mid_mol1, A2)
                    else:
                        leaf_id1 = leaf_type[j1]
                        leaf_id2 = leaf_type[j1]
                        mid_mol1 = combine_leaf_and_core(core_mol, leaf_mol1, L, A1)
                        mid_mol1 = combine_leaf_and_core(mid_mol1, leaf_mol1, L, A2)
                    for j2, leaf_mol2 in enumerate(leaf_mols):
                        if leaf_mol2 == zero_mol:
                            leaf_id3 = 0
                            leaf_id4 = 0
                            mid_mol2 = delete_id(mid_mol1, A3)
                            mid_mol2 = delete_id(mid_mol2, A4)
                        else:
                            leaf_id3 = leaf_type[j2]
                            leaf_id4 = leaf_type[j2]
                            mid_mol2 = combine_leaf_and_core(mid_mol1, leaf_mol2, L, A3)
                            mid_mol2 = combine_leaf_and_core(mid_mol2, leaf_mol2, L, A4)
                        smi = Chem.MolToSmiles(mid_mol2)
                        if smi not in result_smi:
                            result_smi.append(smi)
                            result_types.append(type_id)
                            result_leaves1.append(leaf_id1)
                            result_leaves2.append(leaf_id2)
                            result_leaves3.append(leaf_id3)
                            result_leaves4.append(leaf_id4)
                            result_ids.append(id)

                    if leaf_mol1 == zero_mol:
                        leaf_id1 = 0
                        leaf_id3 = 0
                        mid_mol1 = delete_id(core_mol, A1)
                        mid_mol1 = delete_id(mid_mol1, A3)
                    else:
                        leaf_id1 = leaf_type[j1]
                        leaf_id3 = leaf_type[j1]
                        mid_mol1 = combine_leaf_and_core(core_mol, leaf_mol1, L, A1)
                        mid_mol1 = combine_leaf_and_core(mid_mol1, leaf_mol1, L, A3)
                    for j2, leaf_mol2 in enumerate(leaf_mols):
                        if leaf_mol2 == zero_mol:
                            leaf_id2 = 0
                            leaf_id4 = 0
                            mid_mol2 = delete_id(mid_mol1, A2)
                            mid_mol2 = delete_id(mid_mol2, A4)
                        else:
                            leaf_id2 = leaf_type[j2]
                            leaf_id4 = leaf_type[j2]
                            mid_mol2 = combine_leaf_and_core(mid_mol1, leaf_mol2, L, A2)
                            mid_mol2 = combine_leaf_and_core(mid_mol2, leaf_mol2, L, A4)
                        smi = Chem.MolToSmiles(mid_mol2)
                        if smi not in result_smi:
                            result_smi.append(smi)
                            result_types.append(type_id)
                            result_leaves1.append(leaf_id1)
                            result_leaves2.append(leaf_id2)
                            result_leaves3.append(leaf_id3)
                            result_leaves4.append(leaf_id4)
                            result_ids.append(id)
    else:
        for i, core_mol in enumerate(core_mols):
            point = int(core_sub_points[i])
            type_id = core_type[i]
            id = core_id[i]
            if point == 1:
                for j, leaf_mol in enumerate(leaf_mols):
                    if leaf_mol == zero_mol:
                        leaf_id = leaf_type[0]
                        mid_mol = delete_id(core_mol, A1)
                    else:
                        leaf_id = leaf_type[j]
                        mid_mol = combine_leaf_and_core(core_mol, leaf_mol, L, A1)
                    smi = Chem.MolToSmiles(mid_mol)
                    if smi not in result_smi:
                        result_smi.append(smi)
                        result_types.append(type_id)
                        result_leaves1.append(leaf_id)
                        result_leaves2.append(0)
                        result_leaves3.append(0)
                        result_leaves4.append(0)
                        result_ids.append(id)

            if point == 2:
                for j, leaf_mol in enumerate(leaf_mols):
                    if leaf_mol == zero_mol:
                        leaf_id = 0
                        mid_mol1 = delete_id(core_mol, A1)
                        mid_mol2 = delete_id(mid_mol1, A2)
                    else:
                        leaf_id = leaf_type[j]
                        mid_mol1 = combine_leaf_and_core(core_mol, leaf_mol, L, A1)
                        mid_mol2 = combine_leaf_and_core(mid_mol1, leaf_mol, L, A2)
                    smi = Chem.MolToSmiles(mid_mol2)
                    if smi not in result_smi:
                        result_smi.append(smi)
                        result_types.append(type_id)
                        result_leaves1.append(leaf_id)
                        result_leaves2.append(leaf_id)
                        result_leaves3.append(0)
                        result_leaves4.append(0)
                        result_ids.append(id)

            if point == 3:
                for j, leaf_mol in enumerate(leaf_mols):
                    if leaf_mol == zero_mol:
                        leaf_id = 0
                        mid_mol1 = delete_id(core_mol, A1)
                        mid_mol2 = delete_id(mid_mol1, A2)
                        mid_mol3 = delete_id(mid_mol2, A3)
                    else:
                        leaf_id = leaf_type[j]
                        mid_mol1 = combine_leaf_and_core(core_mol, leaf_mol, L, A1)
                        mid_mol2 = combine_leaf_and_core(mid_mol1, leaf_mol, L, A2)
                        mid_mol3 = combine_leaf_and_core(mid_mol2, leaf_mol, L, A3)
                    smi = Chem.MolToSmiles(mid_mol3)
                    if smi not in result_smi:
                        result_smi.append(smi)
                        result_types.append(type_id)
                        result_leaves1.append(leaf_id)
                        result_leaves2.append(leaf_id)
                        result_leaves3.append(leaf_id)
                        result_leaves4.append(0)
                        result_ids.append(id)
            if point == 4:
                for j, leaf_mol in enumerate(leaf_mols):
                    if leaf_mol == zero_mol:
                        leaf_id = 0
                        mid_mol1 = delete_id(core_mol, A1)
                        mid_mol2 = delete_id(mid_mol1, A2)
                        mid_mol3 = delete_id(mid_mol2, A3)
                        mid_mol4 = delete_id(mid_mol3, A4)
                    else:
                        leaf_id = leaf_type[j]
                        mid_mol1 = combine_leaf_and_core(core_mol, leaf_mol, L, A1)
                        mid_mol2 = combine_leaf_and_core(mid_mol1, leaf_mol, L, A2)
                        mid_mol3 = combine_leaf_and_core(mid_mol2, leaf_mol, L, A3)
                        mid_mol4 = combine_leaf_and_core(mid_mol3, leaf_mol, L, A4)
                    smi = Chem.MolToSmiles(mid_mol4)
                    if smi not in result_smi:
                        result_smi.append(smi)
                        result_types.append(type_id)
                        result_leaves1.append(leaf_id)
                        result_leaves2.append(leaf_id)
                        result_leaves3.append(leaf_id)
                        result_leaves4.append(leaf_id)
                        result_ids.append(id)
    print('Successfully built mol')

    result_data = []
    feature_data['W'] = feature_data['W'].astype(int)
    i = 0
    for smi, result_type, result_leaf1, result_leaf2, result_leaf3, result_leaf4, result_id in (
            zip(result_smi, result_types, result_leaves1, result_leaves2, result_leaves3, result_leaves4, result_ids)):
        rows_to_add = feature_data[feature_data['W'] != ''].copy()
        for index, row in rows_to_add.iterrows():
            row['SMILES'] = smi
            row['leaf1'] = result_leaf1
            row['leaf2'] = result_leaf2
            row['leaf3'] = result_leaf3
            row['leaf4'] = result_leaf4
            row['ID'] = result_id
            result_data.append(row)
        i += 1

    result_df = pd.DataFrame(result_data)
    print(f'Successfully added {i}/{len(result_smi)}')
    # 将结果保存到新的Excel文件
    result_df.to_excel(output_file_path, index=False)


if __name__ == '__main__':
    # mol_combine(mode='非对称')



    core_data = pd.read_excel('mol_link.xlsx', sheet_name='Sheet1')
    core_smiles = core_data['smiles'].tolist()
    core_smiles = [smiles for smiles in core_smiles if isinstance(smiles, str)]
    core_mols = [Chem.MolFromSmiles(smiles) for smiles in core_smiles]
    with open('core_mols.4.16.pkl', 'wb') as f:
        pickle.dump(core_mols, f)



