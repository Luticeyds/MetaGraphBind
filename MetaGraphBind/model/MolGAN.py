import torch
from torch.nn import Module, Linear, Sigmoid, ReLU, Sequential, Tanh, Dropout, ModuleList
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import to_smiles
from rdkit import Chem
import pandas as pd
import joblib
from .Model import Model

# 定义生成器
class GraphGANModel(Module):
    """
    GraphGANModel实现一个图生成对抗网络(GAN)框架。
    它包括生成器、判别器和价值网络，用于分子图的生成与优化。
    """
    def __init__(self, args, vertexes=10, edges=3, nodes=9, embedding_dim=64, decoder_units=[64,128], soft_gumbel_softmax=False,
                 hard_gumbel_softmax=False, value_network_path='net/trans_12.11_new_bro_3.pt'):
        super(GraphGANModel, self).__init__()
        self.args = args
        self.device = args.device
        self.vertexes = vertexes  # 图中节点的数量
        self.edges = edges  # 边的类型数量
        self.nodes = nodes  # 节点的类型数量
        self.embedding_dim = embedding_dim  # 输入嵌入的维度
        self.decoder_units = decoder_units  # 生成器的隐藏层单元数
        self.dropout = Dropout(p=0.5)

        # 构建生成器
        self.generator = self.build_generator().to(self.device)


        # 构建价值网络
        self.value_network = self.load_value_network(value_network_path).to(self.device)

        # 模式标志
        self.soft_gumbel_softmax = soft_gumbel_softmax  # 是否使用软Gumbel Softmax
        self.hard_gumbel_softmax = hard_gumbel_softmax  # 是否使用硬Gumbel Softmax

        # 反向映射规则
        self.x_map = {
            'atomic_num':  [6, 7, 8, 15, 16, 17], # 对应 C、N、O、P、S、Cl 的原子序数
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
        self.e_map = {
            'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
            'stereo': ['STEREONONE', 'STEREOANY'],
            'is_conjugated': [False, True],
        }
        self.features = None
        self.from_excel('data/mol_link.xlsx')

    def build_generator(self):
        """
        构建生成器模型，使用全连接网络 (MLP) 生成边和节点的特征。
        """
        return Sequential(
            Linear(self.embedding_dim, self.decoder_units[0]),
            ReLU(),
            Linear(self.decoder_units[0], self.decoder_units[1]),
            ReLU(),
            Linear(self.decoder_units[1], self.vertexes * self.nodes + self.edges * self.vertexes * self.vertexes),  # 输出节点和边特征
            Sigmoid()  # 输出范围映射到[0, 1]
        )

    def load_value_network(self, value_network_path):
        """
        加载已经训练好的 Model 作为 value_network。
        Args:
            value_network_path (str): 已经保存的 value_network 参数文件路径。
        Returns:
            Model: 加载后的 value_network。
        """
        # 加载训练好的 Model
        value_network = Model(self.args)  # 初始化 Model
        if value_network_path:
            value_network.load_state_dict(torch.load(value_network_path))  # 加载权重
            print(f"Loaded value_network from {value_network_path}")

        # 冻结 value_network 的参数
        for param in value_network.parameters():
            param.requires_grad = False
        return value_network

    def reverse_x_map(self, output, feature_map):
        """
        将生成器输出的 [0, 1] 特征映射回离散特征。

        Args:
            output (torch.Tensor): 生成器输出的特征，形状为 [batch_size, num_nodes, num_features]。
            feature_map (dict): 特征映射字典，键为特征名称，值为对应的离散值列表。

        Returns:
            torch.Tensor: 映射后的特征张量，形状为 [batch_size, num_nodes, num_features]。
        """
        batch_size, num_nodes, num_features = output.shape
        mapped_features = []  # 存储映射后的特征

        # 遍历每个节点
        for node_idx in range(num_nodes):
            node_output = output[:, node_idx, :]  # 取出第 node_idx 个节点的所有特征，形状为 [batch_size, num_features]

            node_mapped_features = []  # 当前节点的映射特征

            # 遍历每个特征
            for feature_idx, (feature_name, values) in enumerate(feature_map.items()):
                # 如果特征值是数值型（int 或 float）
                if isinstance(values[0], (int, float)):
                    # 将离散值列表转换为张量
                    values_tensor = torch.tensor(values, dtype=torch.float).to(self.device)
                    # 生成器输出是 [0, 1] 范围的连续值
                    # 映射到离散的数值集合中
                    scaled_values = node_output[:, feature_idx] * (len(values) - 1)  # 扩展到 [0, len(values)-1]
                    scaled_values = torch.round(scaled_values).long()  # 四舍五入到最近的整数
                    scaled_values = scaled_values.clamp(min=0, max=len(values) - 1)  # 确保值在合法范围内
                    mapped_values = values_tensor[scaled_values]  # 使用 scaled_values 作为索引
                    node_mapped_features.append(mapped_values)

                # 如果特征值是类别型（字符串），进行类别映射
                else:
                    value_to_index = {v: idx for idx, v in enumerate(values)}  # 将字符串类别映射为整数索引
                    num_classes = len(values)

                    # 生成器输出 * 类别数，得到范围 [0, num_classes)
                    scaled_values = (node_output[:, feature_idx] * num_classes)
                    scaled_values = torch.round(scaled_values).long()  # 四舍五入，得到离散的类别索引
                    scaled_values = scaled_values.clamp(max=num_classes - 1)  # 确保不超过类别的最大值

                    node_mapped_features.append(scaled_values)

            # 将当前节点的映射特征拼接在一起
            node_mapped_features = torch.stack(node_mapped_features, dim=1)  # [batch_size, num_features]
            mapped_features.append(node_mapped_features)

        # 将所有节点的映射特征拼接在一起
        mapped_features = torch.stack(mapped_features, dim=0)  # [batch_size, num_nodes, num_features]

        return mapped_features

    def reverse_e_map(self, edges_logits, e_map):
        """
        将生成器输出的边特征 [0, 1] 映射回离散特征。

        Args:
            edges_logits (torch.Tensor): 生成器输出的边特征，形状为 [batch_size, self.edges, self.vertexes, self.vertexes]。
            e_map (dict): 边特征映射字典，键为特征名称，值为对应的离散值列表。

        Returns:
            torch.Tensor: 映射后的边特征张量，形状为 [batch_size, self.edges, self.vertexes, self.vertexes]。
        """
        batch_size, num_edge_types, num_nodes, _ = edges_logits.shape

        # 初始化映射后的边特征张量
        edges_mapped = torch.zeros_like(edges_logits, dtype=torch.float).to(self.device)

        # 遍历每种边类型
        for edge_type_idx in range(num_edge_types):
            # 取出当前边类型的特征
            edge_type_logits = edges_logits[:, edge_type_idx, :, :]  # [batch_size, self.vertexes, self.vertexes]

            # 遍历每条边 (i, j)
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:  # 忽略自环边
                        # 取出当前边的特征值
                        edge_logits = edge_type_logits[:, i, j]  # [batch_size]

                        # 遍历每个特征
                        for feature_idx, (feature_name, values) in enumerate(e_map.items()):
                            # 如果特征值是数值型（int 或 float）
                            if isinstance(values[0], (int, float)):
                                # 将离散值列表转换为张量
                                values_tensor = torch.tensor(values, dtype=torch.float).to(self.device)
                                # 生成器输出是 [0, 1] 范围的连续值
                                # 映射到离散的数值集合中
                                scaled_values = edge_logits * (len(values) - 1)  # 扩展到 [0, len(values)-1]
                                scaled_values = torch.round(scaled_values).long()  # 四舍五入到最近的整数
                                scaled_values = scaled_values.clamp(min=0, max=len(values) - 1)  # 确保值在合法范围内
                                mapped_values = values_tensor[scaled_values]  # 使用 scaled_values 作为索引
                                edges_mapped[:, edge_type_idx, i, j] = mapped_values

                            # 如果特征值是类别型（字符串或其他类型），进行类别映射
                            else:
                                value_to_index = {v: idx for idx, v in enumerate(values)}  # 将类别映射为整数索引
                                num_classes = len(values)

                                # 生成器输出 * 类别数，得到范围 [0, num_classes)
                                scaled_values = (edge_logits * num_classes)
                                scaled_values = torch.round(scaled_values).long()  # 四舍五入，得到离散的类别索引
                                scaled_values = scaled_values.clamp(max=num_classes - 1)  # 确保不超过类别的最大值

                                # 直接存储类别索引，而不是字符串
                                edges_mapped[:, edge_type_idx, i, j] = scaled_values

        return edges_mapped

    def forward(self, embeddings, temperature=1.0):
        """
        前向传播：生成器生成图，判别器和价值网络对图进行判别和奖励评估。
        """
        # 生成器生成边和节点的特征
        logits = self.generator(embeddings)
        nodes_logits = logits[:, :self.vertexes * self.nodes].view(-1, self.vertexes, self.nodes)  # 节点的特征
        edges_logits = logits[:, self.vertexes * self.nodes:].view(-1, self.edges, self.vertexes, self.vertexes)  # 边的特征

        # 离散化特征映射
        nodes_mapped = self.reverse_x_map(nodes_logits, self.x_map)  # 节点特征映射
        edges_mapped = self.reverse_e_map(edges_logits, self.e_map)  # 边特征映射

        # 提取节点特征：节点数量为 self.vertexes，特征维度为 -1（例如 self.nodes）
        node_features = nodes_mapped.view(self.vertexes, -1).to(self.device)  # [num_nodes, node_features]

        # 初始化 current_bonds：记录每个节点当前已有的键数量
        current_bonds = [0] * self.vertexes  # 初始时所有节点的键数为 0

        # 修正 edge_attr 的形状
        edge_attr = edges_mapped.view(-1, self.edges, self.vertexes * self.vertexes).permute(0, 2,
                                                                                             1)  # [batch_size, num_edges, edge_features]
        edge_attr = edge_attr.view(-1, edge_attr.size(-1))  # [num_edges, edge_features]

        # 转换为 torch_geometric.data.Data 格式
        edge_index = []
        self.valid_edge_indices = []  # 记录非自环边的索引

        # 遍历所有节点对，生成边索引
        for i in range(self.vertexes):
            for j in range(self.vertexes):
                # 忽略自环边，并检查是否满足化学连接规则
                if i != j and self.valid_chemical_connection(i, j, node_features, current_bonds, edge_attr):
                    edge_index.append([i, j])  # 添加边
                    self.valid_edge_indices.append(i * self.vertexes + j)  # 记录对应的索引

                    # 更新 current_bonds（每次添加边时，更新节点的键数量）
                    current_bonds[i] += 1
                    current_bonds[j] += 1


        # 构造 edge_index 张量
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)  # [2, num_edges]

        # 过滤 edge_attr，仅保留 valid_edge_indices 对应的特征
        edge_attr = edge_attr[self.valid_edge_indices, :]  # 根据 valid_edge_indices 筛选特征

        # 构建 PyTorch Geometric 的 Data 对象
        outputs_data = Data(
            x=node_features,  # [num_nodes, node_features]
            edge_index=edge_index,  # [2, num_edges]
            edge_attr=edge_attr  # [num_edges, edge_features]
        ).to(self.device)


        # 价值网络：评估生成图的奖励值
        value = self.value_forward(outputs_data, self.features)

        return outputs_data, value


    def value_forward(self, graph, features):
        """
        价值网络的前向传播。
        """
        print(features.shape)
        try:
            smiles = to_smiles(graph)
            mol = Chem.MolFromSmiles(smiles)

            Eu_values = []
            Am_values = []
            i = 0
            while i < 8:
                feature = features[i].unsqueeze(0).to(self.device)
                reward_value = self.value_network(graph, feature)
                if i < 4:
                    Eu_values.append(reward_value)
                else:
                    Am_values.append(reward_value)
                i += 1

            values_list = [a - b for a, b in zip(Am_values, Eu_values)]
            reward_values = torch.stack(values_list).to(self.device)
            max_value = reward_values.max()
            mean_value = reward_values.mean()

            final_value = (max_value * 0.5 + mean_value * 0.5) / 10
            return torch.tensor(0.0, dtype=torch.float, requires_grad=True) if mol is None else final_value.to(self.device)

        except:
            return torch.tensor(0.0, dtype=torch.float, requires_grad=True)

    def sample_z(self, batch_dim):
        """
        生成潜在空间的随机噪声样本。
        """
        return torch.randn((batch_dim, self.embedding_dim)).to(self.device)

    def valid_chemical_connection(self, node_i, node_j, node_features, current_bonds, edge_attr=None):
        """
        判断节点 i 和 j 之间是否可以建立化学键。

        参数：
        - node_i, node_j: 节点索引
        - node_features: 节点特征矩阵，存储原子类型等信息
        - current_bonds: 当前每个原子的键数列表
        - edge_attr: 当前边属性（键类型），可选

        返回：
        - True 如果两个节点可以形成化学键；否则 False
        """
        # 定义化学价和有效原子对
        atomic_valence = {
            6: 4,  # 碳（C）：最大化学价为 4
            7: 3,  # 氮（N）：最大化学价为 3
            8: 2,  # 氧（O）：最大化学价为 2
            15: 5,  # 磷（P）：最大化学价为 5
            16: 6,  # 硫（S）：通常 2，特殊情况下为 4 或 6
            17: 1  # 氯（Cl）：最大化学价为 1
        }
        valid_pairs = {
            (6, 6): [0., 1., 2., 3.],  # C-C 键
            (6, 8): [0., 1.],  # C-O 键
            (6, 7): [0., 1., 3.],  # C-N 键
            (6, 16): [0., 1.],  # C-S 键
            (6, 17): [0.],  # C-Cl 键
            (7, 8): [0.],  # N-O 键
            (7, 16): [0.],  # N-S 键
            (15, 8): [0., 1.],  # P-O 键
        }
        bond_valence = {0.: 1, 1.: 2, 2.: 3, 3.: 1.5}  # 键类型对化学价的影响

        # 获取原子类型
        atom_i = int(self.get_atom_type(node_i, node_features))
        atom_j = int(self.get_atom_type(node_j, node_features))

        # 计算线性索引
        edge_idx = node_i * self.vertexes + node_j

        # 检查是否在 valid_edge_indices 中
        if edge_idx not in self.valid_edge_indices:
            return False  # 边不存在
        edge_idx = self.valid_edge_indices.index(edge_idx)  # 找到在 edge_attr 中的位置

        # 获取边的属性
        bond_type = edge_attr[edge_idx]
        print(bond_type)



        # 默认键类型为单键
        bond_increment = bond_valence.get(bond_type, 1)  # 默认为单键
        print(bond_type, bond_increment)

        # 判断化学价是否超出
        if current_bonds[node_i] + bond_increment > atomic_valence[atom_i]:
            return False
        if current_bonds[node_j] + bond_increment > atomic_valence[atom_j]:
            return False

        # 检查是否为有效的键类型
        if (atom_i, atom_j) in valid_pairs:
            valid_bond_types = valid_pairs[(atom_i, atom_j)]
        elif (atom_j, atom_i) in valid_pairs:
            valid_bond_types = valid_pairs[(atom_j, atom_i)]
        else:
            return False

        if bond_type is not None and bond_type not in valid_bond_types:
            return False

        # 如果满足所有规则，允许建立化学键
        return True

    def distance_constraint(self, node_i, node_j, distance_matrix):
        """
        距离约束函数：根据节点索引和距离矩阵，判断是否满足距离条件。
        """
        dist = distance_matrix[node_i, node_j]
        return 0.5 <= dist <= 2.0  # 仅允许距离在 0.5 Å ~ 2.0 Å 之间

    def get_atom_type(self, node_index, node_features):
        """
        根据节点特征获取原子类型。
        假设 node_features 是 [num_nodes, feature_dim]，某列表示原子类型编码。
        """
        atom_type = node_features[node_index, 0]  # 假设第 0 列存储原子类型
        return atom_type

    def from_excel(self, excel_file):
        scaler = joblib.load('scaler.joblib')
        data = pd.read_excel(excel_file, sheet_name='Sheet3')

        # Fill missing values with 0
        data = data.fillna(0)

        # Standardize supplementary data
        self.features = torch.tensor(scaler.fit_transform(data.iloc[:, 10:38].values), dtype=torch.float).to(self.device)













