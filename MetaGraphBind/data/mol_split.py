import pandas as pd
import os
import re
import joblib
from rdkit import Chem
from rdkit.Chem import RDConfig, FragmentCatalog, BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import from_smiles
from torch.utils.data import Dataset
from typing import Any, Dict, List

x_map: Dict[str, List[Any]] = {
    'atomic_num': list(range(0, 119)),
    'chirality': ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW',
                   'CHI_OTHER', 'CHI_TETRAHEDRAL', 'CHI_ALLENE', 'CHI_SQUAREPLANAR',
                   'CHI_TRIGONALBIPYRAMIDAL', 'CHI_OCTAHEDRAL'],
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'num_radical_electrons': list(range(0, 5)),
    'hybridization': ['UNSPECIFIED', 'S', 'SP', 'SP2', 'SP3', 'SP2D', 'SP3D', 'SP3D2', 'OTHER'],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map: Dict[str, List[Any]] = {
    'bond_type': ['UNSPECIFIED', 'SINGLE', 'DOUBLE', 'TRIPLE', 'QUADRUPLE', 'QUINTUPLE',
                  'HEXTUPLE', 'ONEANDAHALF', 'TWOANDAHALF', 'THREEANDAHALF', 'FOURANDAHALF',
                  'FIVEANDAHALF', 'AROMATIC', 'IONIC', 'HYDROGEN', 'THREECENTER',
                  'DATIVEONE', 'DATIVE', 'DATIVEL', 'DATIVER', 'OTHER', 'ZERO'],
    'stereo': ['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS'],
    'is_conjugated': [False, True],
}

def to_smiles(data: 'torch_geometric.data.Data',
              kekulize: bool = False) -> Any:
    """Converts a :class:`torch_geometric.data.Data` instance to a SMILES
    string.

    Args:
        data (torch_geometric.data.Data): The molecular graph.
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    from rdkit import Chem

    mol = Chem.RWMol()


    assert data.x is not None
    assert data.num_nodes is not None
    assert data.edge_index is not None
    assert data.edge_attr is not None
    for i in range(data.num_nodes):
        atom = Chem.Atom(int(data.x[i, 0]))
        atom.SetChiralTag(Chem.rdchem.ChiralType.values[int(data.x[i, 1])])
        atom.SetFormalCharge(x_map['formal_charge'][int(data.x[i, 3])])
        atom.SetNumExplicitHs(x_map['num_hs'][int(data.x[i, 4])])
        atom.SetNumRadicalElectrons(x_map['num_radical_electrons'][int(
            data.x[i, 5])])
        atom.SetHybridization(Chem.rdchem.HybridizationType.values[int(
            data.x[i, 6])])
        # atom.SetIsAromatic(bool(data.x[i, 7]))
        mol.AddAtom(atom)

    edges = [tuple(i) for i in data.edge_index.t().tolist()]
    visited = set()

    for i in range(len(edges)):
        src, dst = edges[i]
        if tuple(sorted(edges[i])) in visited:
            continue

        bond_type = Chem.BondType.values[int(data.edge_attr[i, 0])]
        mol.AddBond(src, dst, bond_type)

        # Set stereochemistry:
        stereo = Chem.rdchem.BondStereo.values[int(data.edge_attr[i, 1])]
        if stereo != Chem.rdchem.BondStereo.STEREONONE:
            db = mol.GetBondBetweenAtoms(src, dst)
            db.SetStereoAtoms(dst, src)
            db.SetStereo(stereo)

        # Set conjugation:
        is_conjugated = bool(data.edge_attr[i, 2])
        mol.GetBondBetweenAtoms(src, dst).SetIsConjugated(is_conjugated)

        visited.add(tuple(sorted(edges[i])))



    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)


    return smiles

def find_murcko_link_bond(mol):
    """
    Find bonds in the molecule that link to the Murcko scaffold.

    Parameters:
    - mol: RDKit molecule object.

    Returns:
    - link_bond_list: A list of bonds that link to the Murcko scaffold. Each bond is represented as [u, v],
      where u and v are the indices of the atoms at either end of the bond.
    """
    # Get the Murcko scaffold of the molecule
    core = MurckoScaffold.GetScaffoldForMol(mol)

    # Find the atom indices in the original molecule that match the scaffold structure
    scaffold_index = mol.GetSubstructMatch(core)

    # List to store bonds that link to the Murcko scaffold
    link_bond_list = []

    # Get the total number of bonds in the molecule
    num_bonds = mol.GetNumBonds()

    # Iterate over each bond in the molecule
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()

        # Check if the bond links to the scaffold
        link_score = 0
        if u in scaffold_index:
            link_score += 1
        if v in scaffold_index:
            link_score += 1
        if link_score == 1:
            link_bond_list.append([u, v])

    return link_bond_list



def generate_smiles_by_removing_masks(mol, masks):
    """
    Generates new SMILES strings by removing mask matches from the given molecule.

    Parameters:
    - mol: Molecule to be processed.
    - masks: List of lists, where each sublist contains atom indices to be removed.

    Returns:
    - smiles_list: List of new SMILES strings after removing mask matches.
    - removed_parts_list: SMILES strings for the removed parts.
    """
    smiles_list = []
    removed_parts_list = []

    for mask in masks:
        temp_mol = Chem.Mol(mol)  # Create a temporary molecule to manipulate
        Chem.Kekulize(temp_mol)
        editable_mol = Chem.EditableMol(temp_mol)
        temp_mol2 = Chem.Mol(mol)

        # 创建一个映射，标记哪些原子需要替换为 [Am]
        atom_map = {}
        for atom in temp_mol2.GetAtoms():
            idx = atom.GetIdx()
            if idx in mask:
                atom_map[idx] = False  # 掩码原子，不替换
            else:
                atom_map[idx] = True  # 未掩蔽的原子，需要替换为虚拟原子[*]

        # 替换未掩蔽的原子为 [*]
        for atom in temp_mol2.GetAtoms():
            if atom_map[atom.GetIdx()]:
                atom.SetAtomicNum(0)  # 设置为原子序数为 0 的虚拟原子，表示通配符 [*]
                atom.SetIsotope(0)
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(0)
                atom.SetNoImplicit(True)

        edge = []
        for i in range(temp_mol2.GetNumBonds()):
            bond = temp_mol2.GetBondWithIdx(i)
            begin_aid = bond.GetBeginAtomIdx()
            end_aid = bond.GetEndAtomIdx()
            if begin_aid in mask or end_aid in mask:
                edge.append(i)

        if edge:
            re_smi = Chem.MolToSmiles(Chem.PathToSubmol(temp_mol2, edge))
        else:
            re_smi = Chem.MolFragmentToSmiles(temp_mol, atomsToUse=sorted(mask, reverse=True))

        re_smi = re_smi.replace('*', '[Rf]')

        for atom_idx in sorted(mask, reverse=True):
            editable_mol.RemoveAtom(atom_idx)

        cleaned_mol = editable_mol.GetMol()

        # Check if the molecule is inorganic (no carbon atoms)
        if not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]")):
            cleaned_smiles = None
        else:
            try:
                Chem.SanitizeMol(cleaned_mol)
                cleaned_smiles = Chem.MolToSmiles(cleaned_mol)
            except Exception as e:
                cleaned_smiles = None
                err_smiles = Chem.MolToSmiles(cleaned_mol)
                smiles = Chem.MolToSmiles(mol)
                print(f"Sanitization failed: {e}, {smiles}, {err_smiles}")

        smiles_list.append(cleaned_smiles)
        removed_parts_list.append(re_smi)

    return smiles_list, removed_parts_list
def return_leaf_structure(smiles, mode='BRICS'):
    """
    Identifies and returns the leaf structures (substructures) in a molecule based on BRICS or Murcko scaffolding.

    Parameters:
    - smiles: SMILES string representing the molecule.
    - mode: The mode of scaffolding ('BRICS' or 'Murcko').

    Returns:
    - all_substructure_subset: A dictionary containing the substructures and their corresponding bonds.
    """
    m = Chem.MolFromSmiles(smiles)

    # Determine the bonds to break based on the chosen mode
    if mode == 'BRICS':
        # Find BRICS bonds and convert to a list of sets representing bonds
        res = list(BRICS.FindBRICSBonds(m))  # [((1, 2), ('1', '5'))]
        # Find BRICS link bonds
        all_bond = [set(res[i][0]) for i in range(len(res))]
    elif mode == 'Murcko':
        # Find Murcko link bonds
        all_bond = find_murcko_link_bond(m)

    # Initialize dictionary to store substructures
    all_substructure_subset = dict()

    # Find all atoms involved in the identified bonds
    all_atom = []
    for bond in all_bond:
        all_atom = list(set(all_atom + list(bond)))

    if len(all_atom) > 0:
        # If there are atoms involved in the bonds, identify break atoms
        all_break_atom = dict()
        for atom in all_atom:
            break_atom = []
            for bond in all_bond:
                if atom in bond:
                    break_atom += list(set(bond))
            break_atom = [x for x in break_atom if x != atom]
            all_break_atom[atom] = break_atom

        # Initialize dictionaries to store substructures and used atoms
        substrate_idx = dict()
        used_atom = []

        # Iterate through break atoms to build substructures
        for initial_atom_idx, break_atoms_idx in all_break_atom.items():
            if initial_atom_idx not in used_atom:
                neighbor_idx = [initial_atom_idx]
                substrate_idx_i = neighbor_idx
                begin_atom_idx_list = [initial_atom_idx]
                while len(neighbor_idx) != 0:
                    for idx in begin_atom_idx_list:
                        initial_atom = m.GetAtomWithIdx(idx)
                        neighbor_idx = neighbor_idx + [neighbor_atom.GetIdx() for neighbor_atom in
                                                       initial_atom.GetNeighbors()]
                        exlude_idx = all_break_atom[initial_atom_idx] + substrate_idx_i
                        if idx in all_break_atom.keys():
                            exlude_idx = all_break_atom[initial_atom_idx] + substrate_idx_i + all_break_atom[idx]
                        neighbor_idx = [x for x in neighbor_idx if x not in exlude_idx]
                        substrate_idx_i += neighbor_idx
                        begin_atom_idx_list += neighbor_idx
                    begin_atom_idx_list = [x for x in begin_atom_idx_list if x not in substrate_idx_i]
                substrate_idx[initial_atom_idx] = substrate_idx_i
                used_atom += substrate_idx_i
    else:
        # If there are no atoms involved in the bonds, consider the whole molecule as one substructure
        substrate_idx = dict()
        substrate_idx[0] = [x for x in range(m.GetNumAtoms())]

    substructure_mask = []
    atom_mask = []
    for _, substructure in substrate_idx.items():
        substructure_mask.append(substructure)
        atom_mask = atom_mask + substructure

    # Determine the substructure masks based on the specified mode
    smask = substructure_mask

    # Generate SMILES strings for the substructures after removal
    cleaned_smiles, removed_parts_smiles = generate_smiles_by_removing_masks(m, smask)

    # Store substructures and their bonds in the result dictionary
    all_substructure_subset['substructure'] = substrate_idx
    all_substructure_subset['substructure_bond'] = all_bond
    all_substructure_subset['mask'] = smask
    all_substructure_subset['smiles'] = cleaned_smiles
    all_substructure_subset['removed_parts'] = removed_parts_smiles

    return all_substructure_subset


def return_fg_without_c_i_wash(fg_with_c_i, fg_without_c_i):
    """
    Removes redundant carbon atoms from functional groups identified from SMARTS patterns.

    Parameters:
    - fg_with_c_i: List of functional groups with redundant carbon atoms.
    - fg_without_c_i: List of functional groups without redundant carbon atoms.

    Returns:
    - fg_without_c_i_wash: List of functional groups with redundant carbon atoms removed.
    """
    fg_without_c_i_wash = []

    for fg_with_c in fg_with_c_i:
        for fg_without_c in fg_without_c_i:
            if set(fg_without_c).issubset(set(fg_with_c)):
                fg_without_c_i_wash.append(list(fg_without_c))

    return fg_without_c_i_wash


def return_fg_hit_atom(smiles, fg_name_list, fg_with_ca_list, fg_without_ca_list):
    """
    Identifies functional groups in a molecule based on substructure matches and removes redundant matches.

    Parameters:
    - smiles: SMILES string representing the molecule.
    - fg_name_list: List of functional group names.
    - fg_with_ca_list: List of SMARTS patterns for functional groups with redundant carbon atoms.
    - fg_without_ca_list: List of SMARTS patterns for functional groups without redundant carbon atoms.

    Returns:
    - mask_list: List of cleaned functional group atom indices.
    - name_list: List of cleaned functional group names.
    - smiles_list: SMILES string after removing matched functional groups.
    """
    mol = Chem.MolFromSmiles(smiles)
    hit_at = []  # List to store atom indices of matched functional groups
    hit_fg_name = []  # List to store names of matched functional groups
    all_hit_fg_at = []  # List to store all matched functional group atom indices

    # Iterate through the lists of SMARTS patterns
    for i in range(len(fg_with_ca_list)):
        # Find substructure matches for functional groups with redundant carbon atoms
        fg_with_c_i = mol.GetSubstructMatches(fg_with_ca_list[i])
        # Find substructure matches for functional groups without redundant carbon atoms
        fg_without_c_i = mol.GetSubstructMatches(fg_without_ca_list[i])
        # Remove redundant carbon atoms from the matched functional groups
        fg_without_c_i_wash = return_fg_without_c_i_wash(fg_with_c_i, fg_without_c_i)

        # If there are any matches after washing, store them
        if len(fg_without_c_i_wash) > 0:
            hit_at.append(fg_without_c_i_wash)
            hit_fg_name.append(fg_name_list[i])
            all_hit_fg_at += fg_without_c_i_wash

    # Sort the functional group atoms by their length in descending order
    sorted_all_hit_fg_at = sorted(all_hit_fg_at, key=lambda fg: len(fg), reverse=True)

    # Remove smaller functional groups that are part of larger groups
    remain_fg_list = []
    for fg in sorted_all_hit_fg_at:
        if fg not in remain_fg_list:
            if len(remain_fg_list) == 0:
                remain_fg_list.append(fg)
            else:
                i = 0
                for remain_fg in remain_fg_list:
                    if set(fg).issubset(set(remain_fg)):
                        break
                    else:
                        i += 1
                if i == len(remain_fg_list):
                    remain_fg_list.append(fg)

    # Wash the hit functional groups by using the remaining groups, removing small wrongly matched groups
    fg_mask = []
    fg_name = []
    for j in range(len(hit_at)):
        hit_at_wash_j = []
        for fg in hit_at[j]:
            if fg in remain_fg_list:
                hit_at_wash_j.append(fg)
        if len(hit_at_wash_j) > 0:
            for b, hit_fg_b in enumerate(hit_at_wash_j):
                fg_mask.append(hit_fg_b)  # Add functional group atoms to mask
                fg_name.append(hit_fg_name[j])

    # Create a list of new SMILES strings by removing the matched functional groups
    smiles_list = []
    mask_list = []
    name_list = []

    for mask, name in zip(fg_mask, fg_name):
        temp_mol = Chem.Mol(mol)  # Create a temporary molecule to manipulate
        Chem.Kekulize(temp_mol)
        editable_mol = Chem.EditableMol(temp_mol)

        for atom_idx in sorted(mask, reverse=True):
            editable_mol.RemoveAtom(atom_idx)
        cleaned_mol = editable_mol.GetMol()


        # Check if the molecule is inorganic (no carbon atoms)
        if not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]")):
            cleaned_smiles = None
        else:
            try:
                Chem.SanitizeMol(cleaned_mol)
                cleaned_smiles = Chem.MolToSmiles(cleaned_mol)
            except Exception as e:
                cleaned_smiles = None
                print(f"Sanitization failed: {e}")
        smiles_list.append(cleaned_smiles)
        mask_list.append(mask)
        name_list.append(name)

    return mask_list, name_list, smiles_list



def build_mol_graph_data(smilesList, mode='BRICS'):
    """
    Build molecular graph data for BRICS or Murcko substructure decomposition.

    Parameters:
    - smilesList: List of SMILES strings representing molecules.
    - supplement_datas: List of supplementary data for each molecule.
    - labels: List of labels for each molecule.
    - split_index: List of indices indicating the split (e.g., train/test) for each molecule.
    - mode: Substructure decomposition method, either 'BRICS' or 'Murcko'.

    Returns:
    - dataset: A custom dataset object containing the graph data, supplementary data, labels, indices, and substructure masks.
    """
    # Initialize lists to store SMILES data
    smiles_data = []  # Stores original SMILES, substructures, and removed parts
    failed_molecule = []  # Stores SMILES strings that failed to process

    molecule_number = len(smilesList)  # Total number of molecules

    i = 0
    # Iterate through each molecule in the SMILES list
    while i < molecule_number:
        # Decompose the molecule into substructures based on the specified mode
        substructure_dir = return_leaf_structure(smilesList[i][0], mode=mode)
        # Iterate through each substructure mask
        for smask_i, smiles, removed_parts in zip(substructure_dir['mask'], substructure_dir['smiles'], substructure_dir['removed_parts']):
            try:
                # Convert the SMILES string to a graph data object
                if smiles == '' or '.' in smiles:
                    continue
                # Append decomposition results to the SMILES data list
                smiles_data.append({
                    "Original_SMILES": smilesList[i][0],
                    "Substructure_SMILES": smiles,
                    "Removed_Parts": removed_parts
                })

            except:
                # If an error occurs, decrease the molecule count and add the molecule to the failed list
                failed_molecule.append(smilesList[i])
        i += 1
        # Convert the SMILES data to a DataFrame
    smiles_df = pd.DataFrame(smiles_data)

    # Save the DataFrame to an Excel file
    output_excel_path = mode + '.xlsx'
    smiles_df.to_excel(output_excel_path, index=False)
    print(f"Decomposed SMILES data has been saved to {output_excel_path}")

def build_mol_graph_data_for_fg(smilesList):
    # 39 function group config
    fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
    fparams = FragmentCatalog.FragCatParams(1, 6, fName)

    # List of SMARTS patterns for functional groups without connecting atoms (C, H)
    fg_without_ca_smart = [
        '[N;D2]-[C;D3](=O)-[C;D1;H3]', 'C(=O)[O;D1]', 'C(=O)[O;D2]-[C;D1;H3]',
        'C(=O)-[H]', 'C(=O)-[N;D1]', 'C(=O)-[C;D1;H3]', '[N;D2]=[C;D2]=[O;D1]',
        '[N;D2]=[C;D2]=[S;D1]', '[N;D3](=[O;D1])[O;D1]', '[N;R0]=[O;D1]', '[N;R0]-[O;D1]',
        '[N;R0]-[C;D1;H3]', '[N;R0]=[C;D1;H2]', '[N;D2]=[N;D2]-[C;D1;H3]', '[N;D2]=[N;D1]',
        '[N;D2]#[N;D1]', '[C;D2]#[N;D1]', '[S;D4](=[O;D1])(=[O;D1])-[N;D1]',
        '[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]', '[S;D4](=O)(=O)-[O;D1]',
        '[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]', '[S;D4](=O)(=O)-[C;D1;H3]', '[S;D4](=O)(=O)-[Cl]',
        '[S;D3](=O)-[C;D1]', '[S;D2]-[C;D1;H3]', '[S;D1]', '[S;D1]', '[#9,#17,#35,#53]',
        '[C;D4]([C;D1])([C;D1])-[C;D1]',
        '[C;D4](F)(F)F', '[C;D2]#[C;D1;H]', '[C;D3]1-[C;D2]-[C;D2]1', '[O;D2]-[C;D2]-[C;D1;H3]',
        '[O;D2]-[C;D1;H3]', '[O;D1]', '[O;D1]', '[N;D1]', '[N;D1]', '[N;D1]'
    ]

    # Convert the SMARTS patterns to RDKit molecule objects
    fg_without_ca_list = [Chem.MolFromSmarts(smarts) for smarts in fg_without_ca_smart]

    # Get functional groups with connecting atoms from FragmentCatalog
    fg_with_ca_list = [fparams.GetFuncGroup(i) for i in range(39)]

    # Get the names of the functional groups
    fg_name_list = [fg.GetProp('_Name') for fg in fg_with_ca_list]

    # Initialize lists to store SMILES data
    smiles_data = []
    failed_molecule = []  # List to store molecules that fail processing
    molecule_number = len(smilesList)

    i = 0
    while i < molecule_number:


        # Get the atoms and names of the matched functional groups in the molecule
        # fg_mask: List to store masks for functional groups
        # fg_name: List to store names of the functional groups
        fg_mask, fg_name, de_smiles = return_fg_hit_atom(
            smilesList[i][0], fg_name_list, fg_with_ca_list, fg_without_ca_list
        )

        for j, (fg_mask_j, fg_name_j, smiles) in enumerate(zip(fg_mask, fg_name, de_smiles)):
            try:
                # Append the original SMILES, split SMILES, and functional group SMILES to the list
                smiles_data.append({
                    "Original_SMILES": smilesList[i][0],
                    "Split_SMILES": smiles,
                    "Functional_Group_SMILES": fg_name_j
                })

            except:
                # If transformation fails, add to failed list
                failed_molecule.append(smilesList[i])

        i += 1  # Move to the next molecule

        # Convert the SMILES data to a DataFrame
        smiles_df = pd.DataFrame(smiles_data)

    # Save the DataFrame to an Excel file
    output_excel_path = 'fg.xlsx'
    smiles_df.to_excel(output_excel_path, index=False)
    print(f"SMILES data has been saved to {output_excel_path}")

def build_all_mol_graph_data(excel_file, p_d=False):

    # Read data from Excel file
    smiles_list, supplement_data, labels, number = from_excel(excel_file, p_d=p_d)

    # build data for function group
    # build_mol_graph_data_for_fg(smilesList=smiles_list)

    # build data for brics
    build_mol_graph_data(smilesList=smiles_list, mode='BRICS')

    # build data for murcko
    build_mol_graph_data(smilesList=smiles_list, mode='Murcko')




def from_excel(excel_file, p_d=False):
    scaler = StandardScaler()
    scaler = joblib.load('scaler.joblib')
    if not p_d:
        # Read data from Excel file
        data = pd.read_excel(excel_file)
    else:
        data = excel_file

    # Fill missing values with 0
    data = data.fillna(0)

    # Extract sample numbers
    number = torch.tensor(data['number'].values.reshape(-1, 1), dtype=torch.int)

    # Extract SMILES strings and convert to graph data
    smiles_list = data[['SMILES']].values.tolist()

    # Standardize supplementary data
    supplement_data = torch.tensor(scaler.transform(data.iloc[:, 10:38].values), dtype=torch.float)

    # Standardize log_k values
    log_k = data['Value'].values.reshape(-1, 1)
    labels = torch.tensor(log_k, dtype=torch.float)

    return smiles_list, supplement_data, labels, number





if __name__ == '__main__':
    build_all_mol_graph_data('ligand.xlsx')