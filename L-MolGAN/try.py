from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import pickle
import gzip
import joblib
from rdkit import DataStructs
from rdkit import Chem, RDLogger
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

import math
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from typing import List, Any, Dict

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
def from_smiles(mol, with_hydrogen: bool = False,
                kekulize: bool = True) -> 'torch_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """


    RDLogger.DisableLog('rdApp.*')  # type: ignore


    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    xs: List[List[int]] = []
    for atom in mol.GetAtoms():  # type: ignore
        row: List[int] = []
        row.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        row.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        row.append(x_map['degree'].index(atom.GetTotalDegree()))
        row.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        row.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        row.append(x_map['num_radical_electrons'].index(
            atom.GetNumRadicalElectrons()))
        row.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        row.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        row.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(row)

    x = torch.tensor(xs, dtype=torch.float).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():  # type: ignore
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x.to(torch.float), edge_index=edge_index.to(torch.long), edge_attr=edge_attr.to(torch.float))

class ValDataset(Dataset):
    def __init__(self, mols_list):
        """
        Initialize the JessDataset class.

        Parameters:
        - args (Namespace): The arguments containing dataset parameters.
        """
        self.graph_data = []
        self.mols_list = mols_list
        self.supplement_data = None
        self.labels = None
        self.feature_df = pd.read_pickle('data/feature.pkl').fillna(0)
        self.seed = 42
        self.scaler = joblib.load('data/scaler.joblib')
        self.device = 'cuda:0'
        self.train_ratio = 0.8
        self.batch_size = 4096
        self.val_size = 0
        self.val_loader = None
        self.data_split()

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the given index.

        Parameters:
        - idx (int): The index of the sample to retrieve.

        Returns:
        - graph_features (Tensor): The graph features from smiles.
        - supplement_features (Tensor): The supplementary features.
        - label (Tensor): The label of logk1.
        - number (Tensor): The sample number.
        """
        graph_features = self.graph_data[idx].to(self.device)
        supplement_features = self.supplement_data[idx].to(self.device)
        label = self.labels[idx].to(self.device)
        return graph_features, supplement_features, label


    def data_split(self):
        """
        Split the dataset into training, testing, and validation sets.
        """
        self.data_loading()
        self.val_loader = DataLoader(self, batch_size=self.batch_size, shuffle=False)



    def data_loading(self):
        feature_data = torch.tensor(self.scaler.fit_transform(self.feature_df.iloc[:, 10:38].values), dtype=torch.float)
        log_k = torch.tensor(self.feature_df['Value'].values.reshape(-1, 1), dtype=torch.float)
        for mol in self.mols_list:
            data = from_smiles(mol, with_hydrogen=False, kekulize=True).to(self.device)
            self.graph_data.extend([data] * 22)
            if self.supplement_data is not None:
                self.supplement_data = torch.cat((self.supplement_data, feature_data), dim=0)
                self.labels = torch.cat((self.labels, log_k), dim=0)
            else:
                self.supplement_data = feature_data
                self.labels = log_k
def data_combine(mols):
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
                    if mol.GetAtomWithIdx(start_idx).GetTotalNumHs() == 0 or mol.GetAtomWithIdx(
                            end_idx).GetTotalNumHs() == 0:
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

    def mol_combine(single_mols, double_mols, core_mols_file='data/core_mols.4.16.pkl'):
        result_mol = []
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
                    result_mol.append(mid_mol4)

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
                        result_mol.append(mid_mol4)

        return result_mol

    result_mols = []
    for mol in mols:
        single, double = add_rf_to_c_atoms(mol)
        re_mol = mol_combine(single, double)
        result_mols.extend(re_mol)
    return result_mols

def delta_logk_value(mols, model_file='data/model_ft_2_2_6.9_teacher_all.pt'):

    device = 'cuda:0'
    if mols != []:
        dataloader = ValDataset(mols).val_loader
    else:
        return 0.0, 0.0

    model = torch.load(model_file, weights_only=False)
    model.eval()
    model.to(device)
    first_batch = True
    with torch.no_grad():
        for Graph, features, labels in dataloader:
            Graph, features, labels = Graph.to(device), features.to(device), labels.to(device)
            output = model(Graph, features)
            if first_batch:
                y = labels
                out = output
                first_batch = False
            else:
                y = torch.cat((y, labels))
                out = torch.cat((out, output))
    if 'out' in locals():
        Eu_value = out[::2]
        Am_value = out[1::2]
        det_logk = Am_value - Eu_value
        print(Eu_value, Am_value, det_logk)
        max_value = torch.max(det_logk)/10.0
        mean_value = torch.mean(det_logk)/10.0
    else:
        max_value = torch.tensor(0.0, requires_grad=True, device='cuda:0')
        mean_value = torch.tensor(0.0, requires_grad=True, device='cuda:0')
    if max_value <= 1.0:
        return max_value, mean_value
    else:
        print('max:', max_value)
        return 1.0, mean_value

from rdkit import Chem

def filter_molecules_mol(mol_list):
    """
    过滤 RDKit Mol 对象：
    1. 去芳香性（将芳香原子改为非芳香）
    2. 保留 O=、=O、C#N、N#C 结构
    3. 去除无效分子和原子数大于 9 的分子
    4. 去重
    返回处理后的 Mol 对象列表
    """
    filtered_mols = []

    for mol in mol_list:
        smiles = Chem.MolToSmiles(mol)
        # 1
        s = ''.join([c.upper() if c in {'c', 'n', 's', 'o'} else c for c in smiles])
        # 2
        s = s.replace('O=', 'TEMP_O_EQUAL').replace('=O', 'TEMP_EQUAL_O').replace('C#N', 'TEMP_C_N').replace('N#C', 'TEMP_N_C')
        s = s.replace('=', '').replace('#', '')
        s = s.replace('TEMP_O_EQUAL', 'O=').replace('TEMP_EQUAL_O', '=O').replace('TEMP_C_N', 'C#N').replace('TEMP_N_C', 'N#C')

        # 读取分子并跳过无效SMILES
        new_mol = Chem.MolFromSmiles(s)
        if new_mol is None:
            continue

        if new_mol not in filtered_mols:
            filtered_mols.append(new_mol)

    return filtered_mols

mols = [Chem.MolFromSmiles('O=CC(C)CC(C)=O'), Chem.MolFromSmiles('O=CC(C)CC(C)=O')]
# max, mean = delta_logk_value(mols)

smiles_list = ['c1ccccc1', 'CC(=O)O', 'C#N','O=CC(C)CC(C)=O'  , 'O=CC(C)CC(C)=O']
mol_list = [Chem.MolFromSmiles(smi) for smi in smiles_list]

filtered = filter_molecules_mol(mol_list)
smiles = [Chem.MolToSmiles(mol) for mol in filtered]
print(f"保留 {len(filtered)} 个 mol")