from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS
import torch
from torch_geometric.utils import  from_smiles, to_networkx, to_smiles
from torch_geometric.data import Data
from typing import List, Any, Dict
import networkx as nx
from matplotlib import pyplot as plt
from itertools import chain
import torch

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


def from_smiles(smiles: str, with_hydrogen: bool = False,
                kekulize: bool = False) -> 'torch_geometric.data.Data':
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

    mol = Chem.MolFromSmiles(smiles)

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

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)



def to_smiles(data: 'torch_geometric.data.Data',
              kekulize: bool = False) -> Any:
    """Converts a :class:`torch_geometric.data.Data` instance to a SMILES
    string.

    Args:
        data (torch_geometric.data.Data): The molecular graph.
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """

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

def from_smiles_to_brics(mol: Chem.Mol):
    # 使用 BRICS 找到需要断裂的键
    res = list(BRICS.FindBRICSBonds(mol))  # [((atom1, atom2), ('type1', 'type2'))]

    # 初始化断裂键的信息
    break_bonds = [(atom1, atom2) for (atom1, atom2), _ in res]

    # 将分子断裂，生成各个独立的片段
    atom_groups = []
    visited_atoms = set()

    # 遍历所有原子，基于断裂键构建原子团
    for atom_idx in range(mol.GetNumAtoms()):
        if atom_idx not in visited_atoms:
            stack = [atom_idx]
            current_group = set()

            # 使用深度优先搜索 (DFS) 来遍历与当前原子连通的所有原子
            while stack:
                current_atom = stack.pop()
                if current_atom not in visited_atoms:
                    visited_atoms.add(current_atom)
                    current_group.add(current_atom)

                    # 将邻接的原子加入堆栈，忽略断裂的键
                    atom = mol.GetAtomWithIdx(current_atom)
                    for neighbor in atom.GetNeighbors():
                        neighbor_idx = neighbor.GetIdx()
                        if (current_atom, neighbor_idx) not in break_bonds and (
                        neighbor_idx, current_atom) not in break_bonds:
                            stack.append(neighbor_idx)

            atom_groups.append(current_group)

    # 将独立的片段存为 cliques，每个片段是一个原子团
    cliques = [list(group) for group in atom_groups]

    # 生成 `atom2cliques` 映射
    atom2cliques = [[] for _ in range(mol.GetNumAtoms())]
    for idx, clique in enumerate(cliques):
        for atom in clique:
            atom2cliques[atom].append(idx)

    # 通过 BRICS 键来生成 `edge_index`
    edges = set()
    edge_attrs = []

    for (atom1, atom2), _ in res:
        c1, c2 = atom2cliques[atom1][0], atom2cliques[atom2][0]  # 获取两个原子各自所在的团
        if c1 != c2:  # 只添加不同团之间的连接
            # 为边添加特征属性
            bond = mol.GetBondBetweenAtoms(atom1, atom2)
            if bond:
                e = []
                # 键类型特征
                e.append(e_map['bond_type'].index(str(bond.GetBondType())))
                # 立体化学特征
                e.append(e_map['stereo'].index(str(bond.GetStereo())))
                # 是否共轭特征
                e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

                # 添加无向边以及其特征
                edges.add((c1, c2))
                edge_attrs.append(e)
                edges.add((c2, c1))
                edge_attrs.append(e)

    # 将 edges 转换为 tensor 格式
    if len(edges) > 0:
        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # 将 edge_attr 转换为 tensor 格式
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # 生成最终的 `atom2clique` 映射
    rows = [[i] * len(atom2cliques[i]) for i in range(mol.GetNumAtoms())]
    row = torch.tensor(list(chain.from_iterable(rows)))
    col = torch.tensor(list(chain.from_iterable(atom2cliques)))
    atom2clique = torch.stack([row, col], dim=0).to(torch.long)

    return edge_index, atom2clique, len(cliques), edge_attr


def get_atom_features(mol: Any) -> torch.Tensor:
    """从分子中提取原子的特征并返回特征矩阵"""
    atom_features = []
    for atom in mol.GetAtoms():
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
        atom_features.append(row)

    return torch.tensor(atom_features, dtype=torch.float)


def create_brics_data_from_molecule(mol: Any) -> Data:
    # 调用 from_smiles_to_brics 函数
    edge_index, atom2clique, num_cliques, edge_attr = from_smiles_to_brics(mol)

    # 提取原子特征
    atom_features = get_atom_features(mol)  # 假设该函数返回一个形状为 (num_atoms, num_atom_features) 的 tensor

    # 构建节点特征 x，维度为 (num_cliques, num_atom_features)
    num_atom_features = atom_features.shape[1]  # 原子特征的维度
    x = torch.zeros(num_cliques, num_atom_features, dtype=torch.float)  # 为连通团创建特征矩阵

    # 据 atom2clique 填充节点特征
    # 遍历每个团，找到其对应的原子索引
    for clique_idx in range(num_cliques):
        # 找到属于当前团的所有原子
        atoms_in_clique = (atom2clique[1] == clique_idx).nonzero(as_tuple=True)[0]
        # 计算当前团的原子特征之和
        clique_features = atom_features[atoms_in_clique].sum(dim=0)
        x[clique_idx] = clique_features  # 将聚合后的特征赋值给连通团的特征

    # edge_index 和 edge_attr 已经由 tree_decomposition 返回，无需额外构建

    # 创建 Data 对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_cliques=num_cliques,
                atom2clique_index = atom2clique)
    return data

def visualize_with_geometric(data: Data):
    # 将 Data 转换为 NetworkX 图
    G = to_networkx(data, to_undirected=True)

    # 绘制图
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=500, font_size=10)

    # 添加节点特征标签，选择第一个特征
    labels = {i: f"{i}\n{data.x[i][0].item():.2f}" for i in range(data.x.size(0))}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.title('Graph Visualization with PyTorch Geometric')
    plt.show()


