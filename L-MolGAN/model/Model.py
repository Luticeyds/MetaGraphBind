import torch
from torch.nn import Module
from .layers import GCN1, GCN2, MLP, MLP2, MLP_New

class Model(Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.device = args.device
        self.gcn1 = GCN1(args)
        self.mlp = MLP2(args)

    def freeze_layers(self):
        """
        Freeze the layers of the model except the final layers.
        """
        self.gcn1.freeze_layers()
        # self.mlp.freeze_layers()

    def forward(self, Graph, features):
        g = self.gcn1(Graph)
        # Check whether the shapes of g and features match
        if g.size(0) != features.size(0):
            raise ValueError(
                f"Shape mismatch: g.shape[0]={g.size(0)} and features.shape[0]={features.size(0)} must match for concatenation. G = {Graph.size(0)}, F = {features.size(0)}")
        x = torch.cat((g, features), dim=1).float()
        out = self.mlp(x)
        return out

class TransModel(Module):
    def __init__(self, args):
        super(TransModel, self).__init__()
        self.args = args
        self.device = args.device
        self.gcn1 = GCN1(args)
        self.mlp = MLP_New(args)

    def freeze_layers(self):
        """
        Freeze the layers of the model except the final layers.
        """
        self.gcn1.freeze_layers()
        # self.mlp.freeze_layers()

    def forward(self, Graph, features):
        g = self.gcn1(Graph)
        x = torch.cat((g, features), dim=1).float()
        out = self.mlp(x)
        return out

