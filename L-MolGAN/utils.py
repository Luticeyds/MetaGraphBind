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
from torch_geometric import warnings
from torch_geometric.nn import GATConv
def fix_old_gatconv(model):
    for m in model.modules():
        if isinstance(m, GATConv) and not hasattr(m, "res"):
            m.res = None

from model.Model import Model


NP_model = pickle.load(gzip.open('data/NP_score.pkl.gz'))
SA_model = {i[j]: float(i[0]) for i in pickle.load(gzip.open('data/SA_score.pkl.gz')) for j in range(1, len(i))}

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


class MolecularMetrics(object):

    @staticmethod
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
            try:
                smiles = Chem.MolToSmiles(mol)
            except:
                continue

            # 1
            s = ''.join([c.upper() if c in {'c', 'n', 's', 'o'} else c for c in smiles])
            # 2
            s = s.replace('O=', 'TEMP_O_EQUAL').replace('=O', 'TEMP_EQUAL_O').replace('C#N', 'TEMP_C_N').replace('N#C',
                                                                                                                 'TEMP_N_C')
            s = s.replace('=', '').replace('#', '')
            s = s.replace('TEMP_O_EQUAL', 'O=').replace('TEMP_EQUAL_O', '=O').replace('TEMP_C_N', 'C#N').replace(
                'TEMP_N_C', 'N#C')
            # 读取分子并跳过无效SMILES
            new_mol = Chem.MolFromSmiles(s)
            if new_mol is None:
                continue
            if new_mol not in filtered_mols:
                filtered_mols.append(new_mol)
        return filtered_mols

    @staticmethod
    def delta_logk_value(mols, model_file='data/model_ft_2_2_6.9_teacher_all.pt'):

        device = 'cuda:0'

        data = MolecularMetrics.filter_molecules_mol(mols)
        if len(data) != 0:
            dataloader = ValDataset(data).val_loader
        else:
            return 0.0, 0.0

        model = torch.load(model_file, weights_only=False)
        fix_old_gatconv(model)
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
            max_value = torch.max(det_logk).item()/20.0
            mean_value = torch.mean(det_logk).item()/20.0

        else:
            return 0.0, 0.0
        if max_value <= 1.0:
            return max_value, mean_value
        else:
            print('max:', max_value*20)
            return 1.0, mean_value



    @staticmethod
    def _avoid_sanitization_error(op):
        try:
            return op()
        except ValueError:
            return None

    @staticmethod
    def remap(x, x_min, x_max):
        return (x - x_min) / (x_max - x_min)

    @staticmethod
    def valid_lambda(x):
        return x is not None and Chem.MolToSmiles(x) != ''

    @staticmethod
    def valid_lambda_special(x):
        s = Chem.MolToSmiles(x) if x is not None else ''
        return x is not None and '*' not in s and '.' not in s and s != ''

    @staticmethod
    def valid_scores(mols):

        return np.array(list(map(MolecularMetrics.valid_lambda_special, mols)), dtype=np.float32)

    @staticmethod
    def valid_filter(mols):
        return list(filter(MolecularMetrics.valid_lambda, mols))

    @staticmethod
    def valid_total_score(mols):
        return np.array(list(map(MolecularMetrics.valid_lambda, mols)), dtype=np.float32).mean()

    @staticmethod
    def novel_scores(mols, data):
        return np.array(
            list(map(lambda x: MolecularMetrics.valid_lambda(x) and Chem.MolToSmiles(x) not in data.smiles, mols)))

    @staticmethod
    def novel_filter(mols, data):
        return list(filter(lambda x: MolecularMetrics.valid_lambda(x) and Chem.MolToSmiles(x) not in data.smiles, mols))

    @staticmethod
    def novel_total_score(mols, data):
        valid_mols = MolecularMetrics.valid_filter(mols)

        # 如果 valid_mols 为空，返回默认值或适当处理
        if len(valid_mols) == 0:
            return np.nan  # 或者选择其他处理方法，如返回 0 或直接跳过
        else:
            # 计算 novel_scores，并确保无效值处理
            novel_scores = MolecularMetrics.novel_scores(valid_mols, data)

            # 如果 novel_scores 中有 NaN 或 inf，进行处理
            novel_scores = np.nan_to_num(novel_scores, nan=np.nan, posinf=np.inf, neginf=-np.inf)

            # 计算均值，忽略 NaN 值
            return np.nanmean(novel_scores)

    @staticmethod
    def unique_scores(mols):
        smiles = list(map(lambda x: Chem.MolToSmiles(x) if MolecularMetrics.valid_lambda(x) else '', mols))
        return np.clip(
            0.75 + np.array(list(map(lambda x: 1 / smiles.count(x) if x != '' else 0, smiles)), dtype=np.float32), 0, 1)

    @staticmethod
    def unique_total_score(mols):
        v = MolecularMetrics.valid_filter(mols)
        s = set(map(lambda x: Chem.MolToSmiles(x), v))
        return 0 if len(v) == 0 else len(s) / len(v)





    @staticmethod
    def water_octanol_partition_coefficient_scores(mols, norm=False):
        scores = [MolecularMetrics._avoid_sanitization_error(lambda: Crippen.MolLogP(mol)) if mol is not None else None
                  for mol in mols]
        scores = np.array(list(map(lambda x: -3 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, -2.12178879609, 6.0429063424), 0.0, 1.0) if norm else scores

        return scores

    @staticmethod
    def _compute_SAS(mol):
        fp = Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        score1 = 0.
        nf = 0
        # for bitId, v in fps.items():
        for bitId, v in fps.items():
            nf += v
            sfp = bitId
            score1 += SA_model.get(sfp, -4) * v
        score1 /= nf

        # features score
        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(
            mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nSpiro = Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgeheads = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = nAtoms ** 1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.

        # ---------------------------------------
        # This differs from the paper, which defines:
        #  macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = 0. - sizePenalty - stereoPenalty - \
                 spiroPenalty - bridgePenalty - macrocyclePenalty

        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * .5

        sascore = score1 + score2 + score3

        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11. - (sascore - min + 1) / (max - min) * 9.
        # smooth the 10-end
        if sascore > 8.:
            sascore = 8. + math.log(sascore + 1. - 9.)
        if sascore > 10.:
            sascore = 10.0
        elif sascore < 1.:
            sascore = 1.0

        return sascore

    @staticmethod
    def synthetic_accessibility_score_scores(mols, norm=False):
        scores = [MolecularMetrics._compute_SAS(mol) if mol is not None else None for mol in mols]
        scores = np.array(list(map(lambda x: 10 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, 5, 1.5), 0.0, 1.0) if norm else scores

        return scores

    @staticmethod
    def diversity_scores(mols, data):
        rand_mols = np.random.choice(data.data, 100)
        fps = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048) for mol in rand_mols]

        scores = np.array(
            list(map(lambda x: MolecularMetrics.__compute_diversity(x, fps) if x is not None else 0, mols)))
        scores = np.clip(MolecularMetrics.remap(scores, 0.9, 0.945), 0.0, 1.0)

        return scores

    @staticmethod
    def __compute_diversity(mol, fps):
        ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
        dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
        score = np.mean(dist)
        return score

    @staticmethod
    def drugcandidate_scores(mols, data):

        scores = (MolecularMetrics.constant_bump(
            MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True), 0.210,
            0.945) + MolecularMetrics.synthetic_accessibility_score_scores(mols,
                                                                           norm=True) + MolecularMetrics.novel_scores(
            mols, data) + (1 - MolecularMetrics.novel_scores(mols, data)) * 0.3) / 4

        return scores

    @staticmethod
    def constant_bump(x, x_low, x_high, decay=0.025):
        return np.select(condlist=[x <= x_low, x >= x_high],
                         choicelist=[np.exp(- (x - x_low) ** 2 / decay),
                                     np.exp(- (x - x_high) ** 2 / decay)],
                         default=np.ones_like(x))






def mols2grid_image(mols, molsPerRow):
    mols = [e if e is not None else Chem.RWMol() for e in mols]

    for mol in mols:
        AllChem.Compute2DCoords(mol)

    return Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=(150, 150))


def classification_report(data, model, session, sample=False):
    _, _, _, a, x, _, f, _, _ = data.next_validation_batch()

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
        model.nodes_argmax, model.edges_argmax], feed_dict={model.edges_labels: a, model.nodes_labels: x,
                                                            model.node_features: f, model.training: False,
                                                            model.variational: False})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    y_true = e.flatten()
    y_pred = a.flatten()
    target_names = [str(Chem.rdchem.BondType.values[int(e)]) for e in data.bond_decoder_m.values()]

    print('######## Classification Report ########\n')
    print(sk_classification_report(y_true, y_pred, labels=list(range(len(target_names))),
                                   target_names=target_names))

    print('######## Confusion Matrix ########\n')
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(target_names)))))

    y_true = n.flatten()
    y_pred = x.flatten()
    target_names = [Chem.Atom(e).GetSymbol() for e in data.atom_decoder_m.values()]

    print('######## Classification Report ########\n')
    print(sk_classification_report(y_true, y_pred, labels=list(range(len(target_names))),
                                   target_names=target_names))

    print('\n######## Confusion Matrix ########\n')
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(target_names)))))


def reconstructions(data, model, session, batch_dim=10, sample=False):
    m0, _, _, a, x, _, f, _, _ = data.next_train_batch(batch_dim)

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
        model.nodes_argmax, model.edges_argmax], feed_dict={model.edges_labels: a, model.nodes_labels: x,
                                                            model.node_features: f, model.training: False,
                                                            model.variational: False})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    m1 = np.array([e if e is not None else Chem.RWMol() for e in [data.matrices2mol(n_, e_, strict=True)
                                                                  for n_, e_ in zip(n, e)]])

    mols = np.vstack((m0, m1)).T.flatten()

    return mols


def samples(data, model, session, embeddings, sample=False):
    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
        model.nodes_argmax, model.edges_argmax], feed_dict={
        model.embeddings: embeddings, model.training: False})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    return mols

smi_list = []
def all_scores(mols, data, i, norm=False):
    max_logk, mean_logk = MolecularMetrics.delta_logk_value(mols)
    m0 = {k: list(filter(lambda e: e is not None, v)) for k, v in {
        'SA score': MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=norm),
        'diversity score': MolecularMetrics.diversity_scores(mols, data),
        'max delta logk1 score': [max_logk],
        'mean delta logk1 score': [mean_logk]}.items()}

    m1 = {'valid score': MolecularMetrics.valid_total_score(mols) * 100,
          'unique score': MolecularMetrics.unique_total_score(mols) * 100,
          'novel score': MolecularMetrics.novel_total_score(mols, data) * 100}

    valid_molecules = list(filter(MolecularMetrics.valid_lambda_special, mols))
    txt_file = 'gdb/gdb_1.28.txt'
    all_file = 'gdb/gdb_all_1.28.txt'
    # 使用 set 去重
    unique_molecules = set(map(lambda x: Chem.MolToSmiles(x), valid_molecules))
    with open(txt_file, 'a') as f:
        f.write(str(i) + "\n")
        for smi in unique_molecules:
            if smi not in smi_list:
                smi_list.append(smi)
                f.write(smi + "\n")
    with open(all_file, 'a') as f:
        f.write(str(i) + "\n")
        for smi in unique_molecules:
            f.write(smi + "\n")

    return m0, m1


if __name__ == "__main__":
    if 'out' in locals():
        print('1')