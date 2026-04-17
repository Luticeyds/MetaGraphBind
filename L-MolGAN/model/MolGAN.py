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

# Define generator
class GraphGANModel(Module):
    """
    The GraphGANModel implements a graph generative adversarial network (GAN) framework.
    It includes a generator, a discriminator, and a value network for the generation and optimization of molecular graphs.
    """
    def __init__(self, args, vertexes=10, edges=3, nodes=9, embedding_dim=64, decoder_units=[64,128], soft_gumbel_softmax=False,
                 hard_gumbel_softmax=False, value_network_path='net/trans_12.11_new_bro_3.pt'):
        super(GraphGANModel, self).__init__()
        self.args = args
        self.device = args.device
        self.vertexes = vertexes  # the number of nodes in the graph
        self.edges = edges  # the number of edge types
        self.nodes = nodes  # the number of node types
        self.embedding_dim = embedding_dim  # the dimension of the input embedding
        self.decoder_units = decoder_units  # the number of hidden units in the generator
        self.dropout = Dropout(p=0.5)

        # build the generator
        self.generator = self.build_generator().to(self.device)


        # build the value network
        self.value_network = self.load_value_network(value_network_path).to(self.device)

        # mode flag
        self.soft_gumbel_softmax = soft_gumbel_softmax  # whether to use soft Gumbel-Softmax
        self.hard_gumbel_softmax = hard_gumbel_softmax  # whether to use hard Gumbel-Softmax

        # reverse mapping rule
        self.x_map = {
            'atomic_num':  [6, 7, 8, 15, 16, 17], # the atomic numbers corresponding to C, N, O, P, S, and Cl
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
        Build the generator model, using a multilayer perceptron (MLP) to generate edge and node features.
        """
        return Sequential(
            Linear(self.embedding_dim, self.decoder_units[0]),
            ReLU(),
            Linear(self.decoder_units[0], self.decoder_units[1]),
            ReLU(),
            Linear(self.decoder_units[1], self.vertexes * self.nodes + self.edges * self.vertexes * self.vertexes),  # Output node and edge features
            Sigmoid()  # The output range is mapped to [0, 1]
        )

    def load_value_network(self, value_network_path):
        """
        Load a pretrained model as the value_network.
        Args:
            value_network_path (str): Path to the saved parameter file of the value_network.
        Returns:
            Model: The loaded value_network.
        """
        # load the trained model
        value_network = Model(self.args)  # initialize the model
        if value_network_path:
            value_network.load_state_dict(torch.load(value_network_path))  # load the weights
            print(f"Loaded value_network from {value_network_path}")

        # freeze the parameters of the value_network
        for param in value_network.parameters():
            param.requires_grad = False
        return value_network

    def reverse_x_map(self, output, feature_map):
        """
        Map the [0, 1] features output by the generator back to discrete features.

        Args:
            output (torch.Tensor): Features output by the generator, with shape [batch_size, num_nodes, num_features].
            feature_map (dict): A feature mapping dictionary, where the keys are feature names and the values are the corresponding lists of discrete values.

        Returns:
            torch.Tensor: The mapped feature tensor, with shape [batch_size, num_nodes, num_features].
        """

        batch_size, num_nodes, num_features = output.shape
        mapped_features = []  # store the mapped features

        # iterate over each node
        for node_idx in range(num_nodes):
            node_output = output[:, node_idx, :]  # Extract all features of the node_idx-th node, with shape [batch_size, num_features].

            node_mapped_features = []  # the mapped features of the current node

            # iterate over each feature
            for feature_idx, (feature_name, values) in enumerate(feature_map.items()):
                # if the feature value is numeric (int or float)
                if isinstance(values[0], (int, float)):
                    # convert the list of discrete values into a tensor
                    values_tensor = torch.tensor(values, dtype=torch.float).to(self.device)
                    # The generator outputs continuous values in the range [0, 1].
                    # map to a set of discrete values
                    scaled_values = node_output[:, feature_idx] * (len(values) - 1)  # scale to the range [0, len(values) - 1]
                    scaled_values = torch.round(scaled_values).long()  # round to the nearest integer
                    scaled_values = scaled_values.clamp(min=0, max=len(values) - 1)  # ensure the value is within the valid range
                    mapped_values = values_tensor[scaled_values]  # use scaled_values as indices
                    node_mapped_features.append(mapped_values)

                # If the feature value is categorical (string), perform category mapping.
                else:
                    value_to_index = {v: idx for idx, v in enumerate(values)}  # map string categories to integer indices
                    num_classes = len(values)

                    # Multiply the generator output by the number of classes to obtain a range of [0, num_classes).
                    scaled_values = (node_output[:, feature_idx] * num_classes)
                    scaled_values = torch.round(scaled_values).long()  # Round to obtain discrete class indices.
                    scaled_values = scaled_values.clamp(max=num_classes - 1)  # ensure it does not exceed the maximum class value

                    node_mapped_features.append(scaled_values)

            # concatenate the mapped features of the current node together
            node_mapped_features = torch.stack(node_mapped_features, dim=1)  # [batch_size, num_features]
            mapped_features.append(node_mapped_features)

        # concatenate the mapped features of all nodes together
        mapped_features = torch.stack(mapped_features, dim=0)  # [batch_size, num_nodes, num_features]

        return mapped_features

    def reverse_e_map(self, edges_logits, e_map):
        """
        Map the [0, 1] edge features output by the generator back to discrete features.

        Args:
            edges_logits (torch.Tensor): Edge features output by the generator, with shape [batch_size, self.edges, self.vertexes, self.vertexes].
            e_map (dict): An edge feature mapping dictionary, where the keys are feature names and the values are the corresponding lists of discrete values.

        Returns:
            torch.Tensor: The mapped edge feature tensor, with shape [batch_size, self.edges, self.vertexes, self.vertexes].
        """
        batch_size, num_edge_types, num_nodes, _ = edges_logits.shape

        # initialize the mapped edge feature tensor
        edges_mapped = torch.zeros_like(edges_logits, dtype=torch.float).to(self.device)

        # iterate over each edge type
        for edge_type_idx in range(num_edge_types):
            # extract the features of the current edge type
            edge_type_logits = edges_logits[:, edge_type_idx, :, :]  # [batch_size, self.vertexes, self.vertexes]

            # iterate over each edge (i, j)
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:  # ignore self-loop edges
                        # extract the feature value of the current edge
                        edge_logits = edge_type_logits[:, i, j]  # [batch_size]

                        # iterate over each feature
                        for feature_idx, (feature_name, values) in enumerate(e_map.items()):
                            # if the feature value is numeric (int or float)
                            if isinstance(values[0], (int, float)):
                                # convert the list of discrete values into a tensor
                                values_tensor = torch.tensor(values, dtype=torch.float).to(self.device)
                                # The output of the generator is a continuous value within the range of [0, 1]
                                # Map to a discrete set of numerical values
                                scaled_values = edge_logits * (len(values) - 1)  # extended to [0, len(values)-1]
                                scaled_values = torch.round(scaled_values).long()  # Round to the nearest integer
                                scaled_values = scaled_values.clamp(min=0, max=len(values) - 1)  # Ensure that the value is within the legal range
                                mapped_values = values_tensor[scaled_values]  # Use scaled_values as the index
                                edges_mapped[:, edge_type_idx, i, j] = mapped_values

                            # If the feature value is of a categorical type (string or other types), perform categorical mapping
                            else:
                                value_to_index = {v: idx for idx, v in enumerate(values)}  # Map categories to integer indices
                                num_classes = len(values)

                                # The generator outputs * the number of classes, resulting in a range of [0, num_classes)
                                scaled_values = (edge_logits * num_classes)
                                scaled_values = torch.round(scaled_values).long()  # Rounding to the nearest whole number, we obtain the discrete category index
                                scaled_values = scaled_values.clamp(max=num_classes - 1)  # Ensure that the maximum value for the category is not exceeded

                                # Store category indexes directly, rather than strings
                                edges_mapped[:, edge_type_idx, i, j] = scaled_values

        return edges_mapped

    def forward(self, embeddings, temperature=1.0):
        """
        Forward propagation: The generator generates a graph, and the discriminator and value network perform discrimination and reward evaluation on the graph.
        """
        # The generator produces features for edges and nodes
        logits = self.generator(embeddings)
        nodes_logits = logits[:, :self.vertexes * self.nodes].view(-1, self.vertexes, self.nodes)  # Node features
        edges_logits = logits[:, self.vertexes * self.nodes:].view(-1, self.edges, self.vertexes, self.vertexes)  # Edge features

        # Discrete feature mapping
        nodes_mapped = self.reverse_x_map(nodes_logits, self.x_map)  # Node feature mapping
        edges_mapped = self.reverse_e_map(edges_logits, self.e_map)  # edge feature mapping

        # Extract node features: The number of nodes is self.vertexes, and the feature dimension is -1 (e.g., self.nodes)
        node_features = nodes_mapped.view(self.vertexes, -1).to(self.device)  # [num_nodes, node_features]

        # Initialize current_bonds: Record the current number of keys owned by each node
        current_bonds = [0] * self.vertexes  # Initially, the key count of all nodes is 0

        # Correct the shape of edge_attr
        edge_attr = edges_mapped.view(-1, self.edges, self.vertexes * self.vertexes).permute(0, 2,
                                                                                             1)  # [batch_size, num_edges, edge_features]
        edge_attr = edge_attr.view(-1, edge_attr.size(-1))  # [num_edges, edge_features]

        # Convert to torch_geometric.data.Data format
        edge_index = []
        self.valid_edge_indices = []  # Record the index of non-self-loop edges

        # Traverse all node pairs and generate edge indices
        for i in range(self.vertexes):
            for j in range(self.vertexes):
                # Ignore self-loop edges and check whether the chemical connectivity rules are satisfied
                if i != j and self.valid_chemical_connection(i, j, node_features, current_bonds, edge_attr):
                    edge_index.append([i, j])  # Add edges
                    self.valid_edge_indices.append(i * self.vertexes + j)  # Record the corresponding index

                    # Update current_bonds (update the number of keys for the node each time an edge is added)
                    current_bonds[i] += 1
                    current_bonds[j] += 1


        # Construct the edge_index tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)  # [2, num_edges]

        # Filter edge_attr and retain only the features corresponding to valid_edge_indices
        edge_attr = edge_attr[self.valid_edge_indices, :]  # Filter features based on valid_edge_indices

        # Constructing the Data object of PyTorch Geometric
        outputs_data = Data(
            x=node_features,  # [num_nodes, node_features]
            edge_index=edge_index,  # [2, num_edges]
            edge_attr=edge_attr  # [num_edges, edge_features]
        ).to(self.device)


        # Value network: Evaluating the reward value of generated graphs
        value = self.value_forward(outputs_data, self.features)

        return outputs_data, value


    def value_forward(self, graph, features):
        """
        Forward propagation of the value network.
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
        Generate random noise samples in the latent space.
        """
        return torch.randn((batch_dim, self.embedding_dim)).to(self.device)

    def valid_chemical_connection(self, node_i, node_j, node_features, current_bonds, edge_attr=None):
        """
        Determine whether a chemical bond can be formed between nodes i and j.

        parameter
        - node_i, node_j: node indices
        - node_features: Node feature matrix, storing information such as atom types
        - current_bonds: A list of the current bond counts for each atom
        - edge_attr: current edge attribute (key type), optional

        Return:
        - True if two nodes can form a chemical bond; otherwise False
        """
        # Define chemical valence and effective atomic pair
        atomic_valence = {
            6: 4,  # Carbon (C): The maximum chemical valence is 4
            7: 3,  # Nitrogen (N): The maximum chemical valence is 3
            8: 2,  # Oxygen (O): The maximum chemical valence is 2
            15: 5,  # Phosphorus (P): maximum chemical valence is 5
            16: 6,  # Sulfur (S): usually 2, or 4 or 6 in special cases
            17: 1  # Chlorine (Cl): maximum chemical valence is 1
        }
        valid_pairs = {
            (6, 6): [0., 1., 2., 3.],  # C-C
            (6, 8): [0., 1.],  # C-O
            (6, 7): [0., 1., 3.],  # C-N
            (6, 16): [0., 1.],  # C-S
            (6, 17): [0.],  # C-Cl
            (7, 8): [0.],  # N-O
            (7, 16): [0.],  # N-S
            (15, 8): [0., 1.],  # P-O
        }
        bond_valence = {0.: 1, 1.: 2, 2.: 3, 3.: 1.5}  # The influence of bond type on chemical valence

        # Get atomic type
        atom_i = int(self.get_atom_type(node_i, node_features))
        atom_j = int(self.get_atom_type(node_j, node_features))

        # Calculate linear index
        edge_idx = node_i * self.vertexes + node_j

        # Check if it is in valid_edge_indices
        if edge_idx not in self.valid_edge_indices:
            return False  # Edge does not exist
        edge_idx = self.valid_edge_indices.index(edge_idx)  # Find the position in edge_attr

        # Get the attributes of the edge
        bond_type = edge_attr[edge_idx]
        print(bond_type)



        # The default key type is single key
        bond_increment = bond_valence.get(bond_type, 1)  # 默认为单键
        print(bond_type, bond_increment)

        # Determine whether the chemical valence exceeds the limit
        if current_bonds[node_i] + bond_increment > atomic_valence[atom_i]:
            return False
        if current_bonds[node_j] + bond_increment > atomic_valence[atom_j]:
            return False

        # Check whether it is a valid key type
        if (atom_i, atom_j) in valid_pairs:
            valid_bond_types = valid_pairs[(atom_i, atom_j)]
        elif (atom_j, atom_i) in valid_pairs:
            valid_bond_types = valid_pairs[(atom_j, atom_i)]
        else:
            return False

        if bond_type is not None and bond_type not in valid_bond_types:
            return False

        # If all rules are satisfied, chemical bonding is allowed to form
        return True

    def distance_constraint(self, node_i, node_j, distance_matrix):
        """
        Distance constraint function: Determine whether the distance condition is met based on the node index and distance matrix.
        """
        dist = distance_matrix[node_i, node_j]
        return 0.5 <= dist <= 2.0  # Only distances between 0.5 Å and 2.0 Å are allowed

    def get_atom_type(self, node_index, node_features):
        """
        Obtain the atomic type based on the node features.
        Suppose `node_features` is of shape `[num_nodes, feature_dim]`, where a certain column represents the atomic type encoding.
        """
        atom_type = node_features[node_index, 0]  # Suppose the 0th column stores the atomic type
        return atom_type

    def from_excel(self, excel_file):
        scaler = joblib.load('scaler.joblib')
        data = pd.read_excel(excel_file, sheet_name='Sheet3')

        # Fill missing values with 0
        data = data.fillna(0)

        # Standardize supplementary data
        self.features = torch.tensor(scaler.fit_transform(data.iloc[:, 10:38].values), dtype=torch.float).to(self.device)
