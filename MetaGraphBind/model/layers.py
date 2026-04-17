import torch
from torch.nn import Module, Linear, Sequential, LeakyReLU, Parameter, ModuleList, Tanh, Sigmoid, Softmax
import torch.nn.functional as F
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.nn.pool.connect.filter_edges import filter_adj
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ChebConv, GraphConv, global_mean_pool, global_max_pool


class Uout(Module):
    """
    Uout Regularization Module

    Uout regularization is a technique to prevent neural networks from overfitting.
    It introduces a random perturbation to each element of the input during training,
    enhancing the model's robustness. Specifically, for each input x, it is multiplied by a
    random variable epsilon which is uniformly distributed in the range [1 - beta, 1 + beta].

    Reference:
    https://arxiv.org/pdf/1801.05134

    Parameters:
    - p (float): Perturbation magnitude parameter, default value is 0.1, which corresponds to beta = 0.1.
    """

    def __init__(self, p=0.1):
        super(Uout, self).__init__()
        self.beta = p

    def forward(self, x):
        if self.training:
            epsilon = (torch.rand_like(x) - 0.5) * 2 * self.beta + 1
            return x * epsilon
        else:
            return x

class GSAPool(Module):
    """
    Graph Self-Attention Pooling layer as described in the paper:
    https://dl.acm.org/doi/abs/10.1145/3366423.3380083
    """
    def __init__(self, in_channels, ratio=0.5, alpha=0.6, pooling_conv="GCNConv", fusion_conv="GATConv",
                 min_score=None, multiplier=1, non_linearity=torch.tanh):
        """
        Initialize the GSAPool layer.

        Parameters:
        - in_channels (int): Number of input channels.
        - ratio (float, optional): Ratio of nodes to keep after pooling. Defaults to 0.5.
        - alpha (float, optional): Hyperparameter for combining score_s and score_f. Defaults to 0.6.
        - pooling_conv (str, optional): Type of convolution to use for pooling. Defaults to "GCNConv".
        - fusion_conv (str, optional): Type of convolution to use for feature fusion. Defaults to "GATConv".
        - min_score (float, optional): Minimum score for node selection. If None, use non_linearity. Defaults to None.
        - multiplier (int, optional): Multiplier for the output features. Defaults to 1.
        - non_linearity (callable, optional): Non-linearity function to apply to scores. Defaults to torch.tanh.
        """
        super(GSAPool, self).__init__()
        self.in_channels = in_channels
        # self.ratio = torch.tensor(ratio)
        self.ratio = Parameter(torch.tensor(ratio))
        self.ratio.register_hook(self.clamp_ratio)
        self.alpha = Parameter(torch.tensor(alpha))
        self.alpha.register_hook(self.clamp_alpha)
        self.sbtl_layer = self.conv_selection(pooling_conv, in_channels)
        self.fbtl_layer = Linear(in_channels, 1)
        self.fusion = self.conv_selection(fusion_conv, in_channels, conv_type=1)
        self.min_score = min_score
        self.multiplier = multiplier
        self.fusion_flag = 0
        if fusion_conv != "false":
            self.fusion_flag = 1
        self.non_linearity = non_linearity

    def clamp_ratio(self, grad):
        """Ensure that the ratio parameter stays within the range [0, 1]."""
        with torch.no_grad():
            self.ratio.data = torch.clamp(self.ratio.data, 0, 1)

    def clamp_alpha(self, grad):
        """Ensure that the ratio parameter stays within the range [0, 1]."""
        with torch.no_grad():
            self.alpha.data = torch.clamp(self.alpha.data, 0, 1)

    def conv_selection(self, conv, in_channels, conv_type=0):
        """
        Select and initialize the appropriate convolution layer.

        Parameters:
        - conv (str): Type of convolution layer (e.g., "GCNConv", "GATConv").
        - in_channels (int): Number of input channels.
        - conv_type (int, optional): Type of convolution. 0 for pooling, 1 for fusion. Defaults to 0.

        Returns:
        - Conv Layer: The initialized convolution layer.
        """
        if conv_type == 0:
            out_channels = 1
        elif conv_type == 1:
            out_channels = in_channels
        if conv == "GCNConv":
            return GCNConv(in_channels, out_channels)
        elif conv == "ChebConv":
            return ChebConv(in_channels, out_channels, 1)
        elif conv == "SAGEConv":
            return SAGEConv(in_channels, out_channels)
        elif conv == "GATConv":
            return GATConv(in_channels, out_channels, heads=1, concat=True)
        elif conv == "GraphConv":
            return GraphConv(in_channels, out_channels)
        else:
            raise ValueError("Invalid convolution type specified.")

    def forward(self, x, raw, edge_index, edge_attr=None, batch=None):
        """
        Forward pass for the GSAPool layer.

        Parameters:
        - x (Tensor): Input node features.
        - edge_index (Tensor): Edge indices.
        - edge_attr (Tensor, optional): Edge attributes. Defaults to None.
        - batch (Tensor, optional): Batch vector. Defaults to None.

        Returns:
        - x (Tensor): Pooled node features.
        - edge_index (Tensor): Pooled edge indices.
        - edge_attr (Tensor, optional): Pooled edge attributes.
        - batch (Tensor): Updated batch vector.
        - perm (Tensor): Indices of the selected nodes.
        """
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        # Score-based Topology Learner (SBTL)
        score_s = self.sbtl_layer(x, edge_index).squeeze()

        # Feature-based Topology Learner (FBTL)
        score_f = self.fbtl_layer(x).squeeze()

        # Combine scores with hyperparameter alpha
        score = score_s * self.alpha + score_f * (1 - self.alpha)
        score = score.unsqueeze(-1) if score.dim() == 0 else score

        # Apply non-linearity or softmax to scores
        if self.min_score is None:
            score = self.non_linearity(score)
        else:
            score = F.softmax(score, batch)

        # Select top nodes based on scores
        perm = topk(score, self.ratio, batch)

        # Feature fusion
        if self.fusion_flag == 1:
            x = self.fusion(x, edge_index)

        # Pool features and scale by scores

        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x
        raw = raw[perm]

        batch = batch[perm]

        # Update edge_index and edge_attr
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, raw, edge_index, edge_attr, batch, perm, score

class GCN1(Module):
    def __init__(self, args, freeze_layers = False):
        super(GCN1, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.num_hidden_GCN = args.num_hidden_GCN
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling
        # GCN layers
        self.conv1 = GCNConv(self.num_features, self.num_hidden_GCN)
        self.pool1 = GSAPool(in_channels=self.num_hidden_GCN, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.num_hidden_GCN, self.num_hidden_GCN)
        self.pool2 = GSAPool(in_channels=self.num_hidden_GCN, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.num_hidden_GCN, self.num_hidden_GCN)
        self.pool3 = GSAPool(in_channels=self.num_hidden_GCN, ratio=self.pooling_ratio)
        self.mlp = Sequential(Linear(self.num_hidden_GCN * 2, self.num_hidden_GCN),
                              LeakyReLU(),
                              Linear(self.num_hidden_GCN, self.num_classes))
        if freeze_layers:
            self.freeze_layers()
    def freeze_layers(self):
        """
        Freeze the layers of the model except the final layers.
        """
        unfreeze_layers = ['mlp', 'conv3', 'pool3'] # 'conv3', 'pool3'
        for name, param in self.named_parameters():
            if not any(layer in name for layer in unfreeze_layers):
                param.requires_grad = False

    def forward(self, graph):
        # GCN
        # Extract graph components
        x, edge_index, edge_attr, batch = graph.x, graph.edge_index, graph.edge_attr, graph.batch

        # First GCN layer and pooling
        gcn1 = F.leaky_relu(self.conv1(x, edge_index))
        pool1, raw1, edge_index, edge_attr, batch, perm1, score1 = self.pool1(gcn1, x, edge_index, edge_attr, batch=batch)
        global_pool1 = torch.cat((global_mean_pool(pool1, batch), global_max_pool(pool1, batch)), dim=1)


        # Second GCN layer and pooling
        gcn2 = F.leaky_relu(self.conv2(pool1, edge_index))
        pool2, raw2, edge_index, edge_attr, batch, perm2, score2 = self.pool2(gcn2, raw1, edge_index, edge_attr, batch=batch)
        global_pool2 = torch.cat((global_mean_pool(pool2, batch), global_max_pool(pool2, batch)), dim=1)


        # Third GCN layer and pooling
        gcn3 = F.leaky_relu(self.conv3(pool2, edge_index))
        pool3, raw3, edge_index, edge_attr, batch, perm3, score3 = self.pool3(gcn3, raw2, edge_index, edge_attr, batch=batch)
        global_pool3 = torch.cat((global_mean_pool(pool3, batch), global_max_pool(pool3, batch)), dim=1)

        # Readout and MLP processing
        readout = global_pool1 + global_pool2 + global_pool3
        gcn_out = self.mlp(readout)
        return gcn_out

class GCN2(Module):
    def __init__(self, args, freeze_layers=False):
        super(GCN2, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.num_hidden_GCN = args.num_hidden_GCN
        self.num_classes = args.num_classes
        self.pooling_ratio = 1.0
        # GCN layers
        self.conv1 = GCNConv(self.num_features, self.num_hidden_GCN)
        self.conv2 = GCNConv(self.num_hidden_GCN, self.num_hidden_GCN)
        self.conv3 = GCNConv(self.num_hidden_GCN, self.num_hidden_GCN)
        self.mlp = Sequential(Linear(self.num_hidden_GCN, self.num_hidden_GCN),
                              LeakyReLU(),
                              Linear(self.num_hidden_GCN, self.num_classes))
        if freeze_layers:
            self.freeze_layers()
    def freeze_layers(self):
        """
        Freeze the layers of the model except the final layers.
        """
        unfreeze_layers = ['mlp', 'conv3', 'pool3'] #
        for name, param in self.named_parameters():
            if not any(layer in name for layer in unfreeze_layers):
                param.requires_grad = False

    def forward(self, graph):
        # GCN
        # Extract graph components
        x, edge_index, edge_attr, batch = graph.x, graph.edge_index, graph.edge_attr, graph.batch

        # First GCN layer and pooling
        gcn1 = F.leaky_relu(self.conv1(x, edge_index))
        global_pool1 = global_mean_pool(gcn1, batch)

        # Second GCN layer and pooling
        gcn2 = F.leaky_relu(self.conv2(gcn1, edge_index))
        global_pool2 = global_mean_pool(gcn2, batch)

        # Third GCN layer and pooling
        gcn3 = F.leaky_relu(self.conv3(gcn2, edge_index))
        global_pool3 = global_mean_pool(gcn3, batch)

        # Readout and MLP processing
        readout = global_pool1 + global_pool2 + global_pool3
        gcn_out = self.mlp(readout)
        return gcn_out

class MLP(Module):
    def __init__(self, args, freeze_layers=False):
        """
        Initialize the Model.

        Parameters:
        - args: Argument parser object containing model hyperparameters and configuration.
        - freeze_layers (bool): If True, freeze the weights of certain layers.
        """
        super(MLP, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.num_hidden_GCN = args.num_hidden_GCN
        self.num_classes = args.num_classes

        self.num_layers = args.num_layers
        self.num_hidden_MLP1 = args.num_hidden_MLP1
        self.num_hidden_MLP2 = args.num_hidden_MLP2
        self.num_output = args.num_output
        self.dropout_ratio = args.dropout

        # Input layer
        self.input_layer = Linear(self.num_classes + 28, self.num_hidden_MLP1)
        # Hidden layers
        self.hidden1 = Linear(self.num_hidden_MLP1, self.num_hidden_MLP2)
        self.hidden2 = Linear(self.num_hidden_MLP2, self.num_hidden_MLP2)
        # Output layer
        self.output_layer = Linear(self.num_hidden_MLP2, self.num_output)
        # Auxiliary layers
        self.dropout1 = Uout(0.2)
        self.dropout2 = Uout(self.dropout_ratio)
        # self.bn2 = BatchNorm1d(self.num_hidden_MLP2, track_running_stats=False)

        if freeze_layers:
            self.freeze_layers()

    def freeze_layers(self):
        """
        Freeze the layers of the model except the final layers.
        """
        unfreeze_layers = ['output_layer']
        for name, param in self.named_parameters():
            if not any(layer in name for layer in unfreeze_layers):
                param.requires_grad = False

    def forward(self, x):

        # MLP layers
        x = F.leaky_relu(self.dropout1(self.input_layer(x)))
        x = F.leaky_relu(self.dropout2(self.hidden1(x)))
        for i in range(self.num_layers):
            # x = F.leaky_relu(self.hidden2(self.dropout2(self.hidden2(x)))) # 0
            # x = F.leaky_relu(self.dropout2(self.hidden2(x))) # 2
            x = F.leaky_relu(self.hidden2(self.dropout2(F.leaky_relu(self.hidden2(x)))))  # 3
        out = self.output_layer(x)

        return out

class MLP2(Module):
    def __init__(self, args, freeze_layers=False):
        super(MLP2, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.num_hidden_GCN = args.num_hidden_GCN
        self.num_classes = args.num_classes

        self.num_layers = args.num_layers  # 循环的层数
        self.num_hidden_MLP1 = args.num_hidden_MLP1
        self.num_hidden_MLP2 = args.num_hidden_MLP2
        self.num_output = args.num_output
        self.dropout_ratio = args.dropout

        # Input layer
        self.input_layer = Linear(self.num_classes + 28, self.num_hidden_MLP1)
        # Hidden layers
        self.hidden1 = Linear(self.num_hidden_MLP1, self.num_hidden_MLP2)
        # Create multiple hidden layers
        self.hidden_layers = ModuleList([
            Linear(self.num_hidden_MLP2, self.num_hidden_MLP2) for _ in range(self.num_layers)
        ])
        # Output layer
        self.output_layer = Linear(self.num_hidden_MLP2, self.num_output)
        # Auxiliary layers
        self.dropout1 = Uout(0.2)
        self.dropout2 = Uout(self.dropout_ratio)

        if freeze_layers:
            self.freeze_layers()

    def freeze_layers(self):
        """
        Freeze specified layers of the model.

        Parameters:
        - freeze_indices (list or None): Indices of layers to freeze. If None, no layers are frozen.
        """
        unfreeze_layers = ['output_layer']

        for name, param in self.named_parameters():
            if not any(layer in name for layer in unfreeze_layers):
                param.requires_grad = False

        for i in range(self.num_layers):
            for param in self.hidden_layers[i].parameters():
                param.requires_grad = True

        # 冻结指定的层
        for idx in range(self.num_layers - 3):# 开几层减几
            # 冻结指定的隐藏层
            for param in self.hidden_layers[idx].parameters():
                param.requires_grad = False



    def forward(self, x):
        # MLP layers
        x = F.leaky_relu(self.dropout1(self.input_layer(x)))
        x = F.leaky_relu(self.dropout2(self.hidden1(x)))
        for i, layer in enumerate(self.hidden_layers):
            x = F.leaky_relu(layer(self.dropout2(F.leaky_relu(layer(x)))))
        out = self.output_layer(x)
        return out

class MLP_New(Module):
    def __init__(self, args, freeze_layers=False):
        """
        Initialize the Model.

        Parameters:
        - args: Argument parser object containing model hyperparameters and configuration.
        - freeze_layers (bool): If True, freeze the weights of certain layers.
        """
        super(MLP_New, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.num_hidden_GCN = args.num_hidden_GCN
        self.num_classes = args.num_classes

        self.num_layers = 1
        self.num_hidden_MLP1 = 16
        self.num_hidden_MLP2 = 4
        self.num_output = args.num_output
        self.dropout_ratio = args.dropout

        # Input layer
        self.input_layer = Linear(self.num_classes + 28, self.num_hidden_MLP1)
        # Hidden layers
        self.hidden1 = Linear(self.num_hidden_MLP1, self.num_hidden_MLP2)
        self.hidden2 = Linear(self.num_hidden_MLP2, self.num_hidden_MLP2)
        # Output layer
        self.output_layer = Linear(self.num_hidden_MLP2, self.num_output)
        # Auxiliary layers
        self.dropout1 = Uout(0.2)
        self.dropout2 = Uout(self.dropout_ratio)
        # self.bn2 = BatchNorm1d(self.num_hidden_MLP2, track_running_stats=False)

        if freeze_layers:
            self.freeze_layers()

    def freeze_layers(self):
        """
        Freeze the layers of the model except the final layers.
        """
        unfreeze_layers = ['output_layer']
        for name, param in self.named_parameters():
            if not any(layer in name for layer in unfreeze_layers):
                param.requires_grad = False

    def forward(self, x):

        # MLP layers
        x = F.leaky_relu(self.dropout2(self.input_layer(x)))
        x = F.leaky_relu(self.dropout2(self.hidden1(x)))
        # for i in range(self.num_layers):
        #     # x = F.leaky_relu(self.hidden2(self.dropout2(self.hidden2(x)))) # 0
        #     # x = F.leaky_relu(self.dropout2(self.hidden2(x))) # 2
        #     x = F.leaky_relu(self.hidden2(self.dropout2(F.leaky_relu(self.hidden2(x)))))  # 3
        out = self.output_layer(x)

        return out


