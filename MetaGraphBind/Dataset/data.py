import pandas as pd
import torch
import joblib
import random
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import subgraph
from torch.utils.data import Dataset, random_split, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from .utils import create_brics_data_from_molecule, from_smiles, to_smiles

class PreDataset(Dataset):
    def __init__(self, args):
        """
        Initialize the JessDataset class.

        Parameters:
        - args (Namespace): The arguments containing dataset parameters.
        """
        self.graph_data = None
        self.smiles_list = None
        self.supplement_data = None
        self.labels = None
        self.seed = args.seed
        self.scaler = joblib.load('scaler.joblib')
        self.device = args.device
        self.train_ratio = args.train_ratio
        self.batch_size = args.batch_size
        self.excel_file = args.excel_file
        self.data_from_excel()
        self.train_size = 0
        self.test_size = 0
        self.val_size = 0
        self.splits = None
        self.train_loader = None
        self.test_loader = None
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

    def smiles_to_graph(self):
        """
        Convert the SMILES strings in the list to graph data recognized by torch_geometric.

        Returns:
        - graph_list (list): A list of graph data objects.
        """
        graph_list = []
        for smiles in self.smiles_list:
            data = from_smiles(smiles[0], with_hydrogen=False, kekulize=True).to(self.device)
            graph_list.append(data)

        return graph_list

    def data_split(self):
        """
        Split the dataset into training, testing, and validation sets.
        """
        self.train_size = int(self.train_ratio * len(self))
        self.test_size = (len(self) - self.train_size)//2
        self.val_size = len(self) - self.train_size - self.test_size
        if self.batch_size == 0:
            batch_size_train = self.train_size
            batch_size_test = self.test_size
            batch_size_val = self.val_size
        else:
            batch_size_train = self.batch_size
            batch_size_test = self.batch_size
            batch_size_val = self.batch_size
        train_dataset, temp_dataset = random_split(self, [self.train_size, self.test_size + self.val_size])
        test_dataset, val_dataset = random_split(temp_dataset, [self.test_size, self.val_size])
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)


    def data_from_excel(self):
        """
        Load data from an Excel file into the dataset.
        """
        # Read data from Excel file
        data = pd.read_excel(self.excel_file)

        # Fill missing values with 0
        data = data.fillna(0)

        # Extract SMILES strings and convert to graph data
        self.smiles_list = data[['SMILES']].values.tolist()
        self.graph_data = self.smiles_to_graph()

        # Standardize supplementary data
        self.supplement_data = torch.tensor(self.scaler.fit_transform(data.iloc[:, 10:38].values), dtype=torch.float)
        # Standardize log_k values
        # log_k = self.scaler.fit_transform(data['Value'].values.reshape(-1, 1))
        log_k = data['Value'].values.reshape(-1, 1)
        self.labels = torch.tensor(log_k, dtype=torch.float)

    def split_batch(self, batch):
        """
        Splits a batch of graphs into individual graphs.

        Parameters:
        - batch: A batch of graphs combined into a single graph by DataLoader.

        Returns:
        - List of individual graphs.
        """
        data_list = []
        num_graphs = batch.batch.max().item() + 1

        for i in range(num_graphs):
            mask = (batch.batch == i)
            x = batch.x[mask]
            edge_index, edge_attr = subgraph(mask, batch.edge_index, edge_attr=batch.edge_attr, relabel_nodes=True)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(data)

        return data_list

class FinetuneDataset(Dataset):
    def __init__(self, args):
        """
        Initialize the JessDataset class.

        Parameters:
        - args (Namespace): The arguments containing dataset parameters.
        """
        self.graph_data = None
        self.smiles_list = None
        self.supplement_data = None
        self.labels = None
        self.seed = args.seed
        self.scaler = joblib.load('scaler.joblib')
        self.device = args.device
        self.num_copies = args.num_copies
        self.train_ratio = 0.9
        self.batch_size = args.batch_size
        self.excel_file = args.excel_file
        self.data_from_excel()
        self.train_size = 0
        self.test_size = 0
        self.val_size = 0
        self.splits = None
        self.train_loader = None
        self.test_loader = None
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

    def smiles_to_graph(self):
        """
        Convert the SMILES strings in the list to graph data recognized by torch_geometric.

        Returns:
        - graph_list (list): A list of graph data objects.
        """
        graph_list = []
        for smiles in self.smiles_list:
            data = from_smiles(smiles[0], with_hydrogen=False, kekulize=True).to(self.device)
            graph_list.append(data)


        return graph_list

    def data_split(self):
        """
        Split the dataset into training, testing, and validation sets.
        """
        self.train_size = int(self.train_ratio * len(self))
        self.test_size = (len(self) - self.train_size)//2
        self.val_size = len(self) - self.train_size - self.test_size
        if self.batch_size == 0:
            batch_size_train = self.train_size
            batch_size_test = self.test_size
            batch_size_val = self.val_size
        else:
            batch_size_train = self.batch_size
            batch_size_test = self.batch_size
            batch_size_val = self.batch_size
        train_dataset, temp_dataset = random_split(self, [self.train_size, self.test_size + self.val_size])
        augmented_train_dataset = self.augment_dataset(train_dataset)
        test_dataset, val_dataset = random_split(temp_dataset, [self.test_size, self.val_size])
        self.train_loader = DataLoader(
            augmented_train_dataset,
            batch_size=batch_size_train,
            shuffle=True
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size_test,
            shuffle=False
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size_val,
            shuffle=False
        )



    def data_from_excel(self):
        """
        Load data from an Excel file into the dataset.
        """
        # Read data from Excel file
        data = pd.read_excel(self.excel_file)

        # Fill missing values with 0
        data = data.fillna(0)

        # Extract SMILES strings and convert to graph data
        self.smiles_list = data[['SMILES']].values.tolist()
        self.graph_data = self.smiles_to_graph()

        # Standardize supplementary data
        self.supplement_data = torch.tensor(self.scaler.fit_transform(data.iloc[:, 10:38].values), dtype=torch.float)
        # Standardize log_k values
        # log_k = self.scaler.fit_transform(data['Value'].values.reshape(-1, 1))
        log_k = data['Value'].values.reshape(-1, 1)
        self.labels = torch.tensor(log_k, dtype=torch.float)

    def split_batch(self, batch):
        """
        Splits a batch of graphs into individual graphs.

        Parameters:
        - batch: A batch of graphs combined into a single graph by DataLoader.

        Returns:
        - List of individual graphs.
        """
        data_list = []
        num_graphs = batch.batch.max().item() + 1

        for i in range(num_graphs):
            mask = (batch.batch == i)
            x = batch.x[mask]
            edge_index, edge_attr = subgraph(mask, batch.edge_index, edge_attr=batch.edge_attr, relabel_nodes=True)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(data)

        return data_list

    def augment_dataset(self, dataset):
        """
        对整个训练数据集进行增广。
        Args:
            dataset (Dataset): 原始训练数据集。
            num_copies (int): 每条数据生成的增广版本数量。
        Returns:
            List[Tuple]: 增广后的训练数据列表 (graph_features, supplement_features, label)。
        """
        augmented_data = []
        for graph_features, supplement_features, label in dataset:
            augmented_data.append((graph_features, supplement_features, label))
            if self.num_copies > 0:
                for _ in range(self.num_copies):
                    # 应用增广
                    augmented_graph_features, augmented_supplement_features, augmented_label = self.augment_data(
                        graph_features, supplement_features, label
                    )
                    augmented_data.append((augmented_graph_features, augmented_supplement_features, augmented_label))

        return augmented_data

    def augment_data(self, graph_features, supplement_features, label):
        """
        增广图特征和补充特征。
        Args:
            graph_features: 分子图特征 (torch_geometric.data.Data 或其他图数据格式)。
            supplement_features: 补充特征 (torch.Tensor 或 numpy 数组)。
            label: 标签。
        Returns:
            增广后的 graph_features, supplement_features, label。
        """
        # 增广图特征
        if graph_features is not None:
            graph_features = self.augment_graph(graph_features)

        # 增广补充特征
        if supplement_features is not None:
            supplement_features = self.augment_supplement_features(supplement_features)

        return graph_features, supplement_features, label

    def augment_graph(self, graph):
        """
        对分子图进行增广，包括节点特征、边特征和拓扑结构。
        Args:
            graph (torch_geometric.data.Data): 输入图数据，包含 x, edge_index, edge_attr 等。
        Returns:
            torch_geometric.data.Data: 增广后的图数据。
        """
        # 增广节点特征
        if random.random() > 0.5:
            # 添加高斯噪声
            noise = torch.randn_like(graph.x) * 0.01
            graph.x = graph.x + noise
            # # 随机掩盖部分特征
            # mask = torch.rand(graph.x.size()) > 0.8
            # graph.x = graph.x * mask.float()

        # 增广边特征
        if random.random() > 0.5 and 'edge_attr' in graph:
            # 添加高斯噪声
            graph.edge_attr = graph.edge_attr.float()
            noise = torch.randn_like(graph.edge_attr) * 0.01
            graph.edge_attr = graph.edge_attr + noise

        # # 增广拓扑结构
        # if random.random() > 0.5:
        #     # 随机删除边
        #     num_edges = graph.edge_index.size(1)
        #     keep_edges = torch.rand(num_edges) > 0.1  # 保留90%的边
        #     edge_index = graph.edge_index[:, keep_edges]
        #     edge_attr = graph.edge_attr[keep_edges] if 'edge_attr' in graph else None
        #     # 更新图
        #     graph.edge_index = edge_index
        #     if edge_attr is not None:
        #         graph.edge_attr = edge_attr

        return graph

    def augment_supplement_features(self, features):
        """
        增广补充特征，例如添加噪声或比例缩放。
        Args:
            features: 补充特征 (torch.Tensor)。
        Returns:
            增广后的特征。
        """
        # 添加高斯噪声
        noise = torch.randn_like(features) * 0.01
        features = features + noise


        return features



class ValDataset(Dataset):
    def __init__(self, args):
        """
        Initialize the JessDataset class.

        Parameters:
        - args (Namespace): The arguments containing dataset parameters.
        """
        self.graph_data = None
        self.smiles_list = None
        self.supplement_data = None
        self.labels = None
        self.number = None
        self.seed = args.seed
        self.scaler = joblib.load('scaler.joblib')
        self.device = args.device
        self.train_ratio = 0.8
        self.batch_size = args.batch_size
        self.excel_file = args.excel_file
        self.data_from_excel()
        self.train_size = 0
        self.test_size = 0
        self.val_size = 0
        self.splits = None
        self.train_loader = None
        self.test_loader = None
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

    def smiles_to_graph(self):
        """
        Convert the SMILES strings in the list to graph data recognized by torch_geometric.

        Returns:
        - graph_list (list): A list of graph data objects.
        """
        graph_list = []
        for smiles in self.smiles_list:
            data = from_smiles(smiles[0], with_hydrogen=False, kekulize=True).to(self.device)
            graph_list.append(data)

        return graph_list

    def data_split(self):
        """
        Split the dataset into training, testing, and validation sets.
        """
        self.val_loader = DataLoader(self, batch_size=self.batch_size, shuffle=False)



    def data_from_excel(self):
        """
        Load data from an Excel file into the dataset.
        """
        # Read data from Excel file
        try:
            data = pd.read_excel(self.excel_file)
        except:
            data = pd.read_csv(self.excel_file)

        # Fill missing values with 0
        data = data.fillna(0)

        # Extract sample numbers
        self.number = torch.tensor(data['number'].values.reshape(-1, 1), dtype=torch.int)

        # Extract SMILES strings and convert to graph data
        self.smiles_list = data[['SMILES']].values.tolist()
        self.graph_data= self.smiles_to_graph()

        # Standardize supplementary data
        self.supplement_data = torch.tensor(self.scaler.fit_transform(data.iloc[:, 10:38].values), dtype=torch.float)
        # Standardize log_k values
        # log_k = self.scaler.fit_transform(data['Value'].values.reshape(-1, 1))
        log_k = data['Value'].values.reshape(-1, 1)
        self.labels = torch.tensor(log_k, dtype=torch.float)

class Unlabeled_Dataset(Dataset):
    def __init__(self, args):
        """
        Initialize the JessDataset class.

        Parameters:
        - args (Namespace): The arguments containing dataset parameters.
        """
        self.graph_data = None
        self.brics_data = None
        self.smiles_list = None
        self.supplement_data = None
        self.number = None
        self.seed = args.seed
        self.scaler = joblib.load('scaler.joblib')
        self.device = args.device
        self.batch_size = args.batch_size
        self.excel_file = args.unlabeled_excel_file
        self.data_from_excel()
        self.train_size = 0
        self.test_size = 0
        self.val_size = 0
        self.val_loader = None
        self.data_split()

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.smiles_list)

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
        return graph_features, supplement_features

    def smiles_to_graph(self):
        """
        Convert the SMILES strings in the list to graph data recognized by torch_geometric.

        Returns:
        - graph_list (list): A list of graph data objects.
        """
        graph_list = []
        for smiles in self.smiles_list:
            data = from_smiles(smiles[0], with_hydrogen=False, kekulize=True).to(self.device)
            graph_list.append(data)

        return graph_list

    def data_split(self):
        """
        Split the dataset into training, testing, and validation sets.
        """
        self.val_loader = DataLoader(self, batch_size=self.batch_size, shuffle=False)

    def data_from_excel(self):
        """
        Load data from an Excel file into the dataset.
        """
        # Read data from Excel file
        try:
            data = pd.read_excel(self.excel_file)
        except:
            data = pd.read_csv(self.excel_file)

        # Fill missing values with 0
        data = data.fillna(0)


        # Extract SMILES strings and convert to graph data
        self.smiles_list = data[['SMILES']].values.tolist()
        self.graph_data= self.smiles_to_graph()

        # Standardize supplementary data
        self.supplement_data = torch.tensor(self.scaler.fit_transform(data.iloc[:, 10:38].values), dtype=torch.float)
