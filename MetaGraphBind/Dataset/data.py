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
        self.raw_supplement_data = None
        self.supplement_data = None
        self.labels = None
        self.seed = args.seed
        self.scaler = StandardScaler()
        self.scaler_path = args.pretrain_scaler_path
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
        dataset_size = len(self)
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
        generator = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(dataset_size, generator=generator).tolist()

        self.train_indices = perm[:self.train_size]
        self.test_indices = perm[self.train_size:self.train_size + self.test_size]
        self.val_indices = perm[self.train_size + self.test_size:]

        train_features = self.raw_supplement_data[self.train_indices]
        self.scaler.fit(train_features)
        joblib.dump(self.scaler, self.scaler_path)
        transformed_features = self.scaler.transform(self.raw_supplement_data)
        self.supplement_data = torch.tensor(transformed_features, dtype=torch.float)

        train_dataset = Subset(self, self.train_indices)
        test_dataset = Subset(self, self.test_indices)
        val_dataset = Subset(self, self.val_indices)
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

        # supplementary data
        self.raw_supplement_data = data.iloc[:, 10:38].values.astype(float)

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
    def __init__(self, args, first=True):
        """
        Initialize the JessDataset class.

        Parameters:
        - args (Namespace): The arguments containing dataset parameters.
        """
        self.graph_data = None
        self.smiles_list = None
        self.raw_supplement_data = None
        self.supplement_data = None
        self.labels = None
        self.seed = args.seed
        self.scaler = StandardScaler()
        if first:
            self.scaler_path = args.transfer1_scaler_path
        else:
            self.scaler_path = args.transfer2_scaler_path
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
        dataset_size = len(self)
        self.train_size = int(self.train_ratio * len(self))
        self.test_size = (len(self) - self.train_size) // 2
        self.val_size = len(self) - self.train_size - self.test_size
        if self.batch_size == 0:
            batch_size_train = self.train_size
            batch_size_test = self.test_size
            batch_size_val = self.val_size
        else:
            batch_size_train = self.batch_size
            batch_size_test = self.batch_size
            batch_size_val = self.batch_size
        generator = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(dataset_size, generator=generator).tolist()

        self.train_indices = perm[:self.train_size]
        self.test_indices = perm[self.train_size:self.train_size + self.test_size]
        self.val_indices = perm[self.train_size + self.test_size:]

        train_features = self.raw_supplement_data[self.train_indices]
        self.scaler.fit(train_features)
        joblib.dump(self.scaler, self.scaler_path)
        transformed_features = self.scaler.transform(self.raw_supplement_data)
        self.supplement_data = torch.tensor(transformed_features, dtype=torch.float)

        train_dataset = Subset(self, self.train_indices)
        augmented_train_dataset = self.augment_dataset(train_dataset)
        test_dataset = Subset(self, self.test_indices)
        val_dataset = Subset(self, self.val_indices)
        self.train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size_train, shuffle=True)
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

        # supplementary data
        self.raw_supplement_data = data.iloc[:, 10:38].values.astype(float)
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
        Augment the entire training dataset.

        Args:
            dataset (Dataset): The original training dataset.
            num_copies (int): The number of augmented versions generated for each sample.

        Returns:
            List[Tuple]: The augmented training data list
            (graph_features, supplement_features, label).
        """
        augmented_data = []
        for graph_features, supplement_features, label in dataset:
            augmented_data.append((graph_features, supplement_features, label))
            if self.num_copies > 0:
                for _ in range(self.num_copies):
                    # Apply augmentation
                    augmented_graph_features, augmented_supplement_features, augmented_label = self.augment_data(
                        graph_features, supplement_features, label
                    )
                    augmented_data.append((augmented_graph_features, augmented_supplement_features, augmented_label))

        return augmented_data

    def augment_data(self, graph_features, supplement_features, label):
        """
        Augmented graph features and supplementary features.
        Args:
            graph_features: Molecular graph features (torch_geometric.data.Data or other graph data formats).
            supplement_features: Supplementary features (torch.Tensor or numpy array).
            label: label.
        Returns:
            The augmented graph_features, supplement_features, and label.
        """
        # augmented graph features
        if graph_features is not None:
            graph_features = self.augment_graph(graph_features)

        # augmentation and supplementary features
        if supplement_features is not None:
            supplement_features = self.augment_supplement_features(supplement_features)

        return graph_features, supplement_features, label

    def augment_graph(self, graph):
        """
        Augmenting the molecular graph involves enhancing its node features, edge features, and topological structure.
        Args:
            graph (torch_geometric.data.Data): Input graph data, including x, edge_index, edge_attr, etc.
        Returns:
            torch_geometric.data.Data: Augmented graph data.
        """
        # Augmenting node features
        if random.random() > 0.5:
            # Add Gaussian noise
            noise = torch.randn_like(graph.x) * 0.01
            graph.x = graph.x + noise

        # augmented edge features
        if random.random() > 0.5 and 'edge_attr' in graph:
            # Add Gaussian noise
            graph.edge_attr = graph.edge_attr.float()
            noise = torch.randn_like(graph.edge_attr) * 0.01
            graph.edge_attr = graph.edge_attr + noise

        return graph

    def augment_supplement_features(self, features):
        """
        Augmenting supplementary features, such as adding noise or scaling.
        Args:
            features: Supplementary features (torch.Tensor).
        Returns:
            Augmented features.
        """
        # Add Gaussian noise
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
        self.scaler = joblib.load(args.transfer2_scaler_path)
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
        self.supplement_data = torch.tensor(self.scaler.transform(data.iloc[:, 10:38].values), dtype=torch.float)
        log_k = data['Value'].values.reshape(-1, 1)
        self.labels = torch.tensor(log_k, dtype=torch.float)


class Unlabeled_Dataset(Dataset):
    def __init__(self, args, first=True):
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
        if first:
            self.scaler = joblib.load(args.transfer1_scaler_path)
        else:
            self.scaler = joblib.load(args.transfer2_scaler_path)

        self.device = args.device
        self.batch_size = args.batch_size
        self.excel_file = args.generated_excel
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
        self.supplement_data = torch.tensor(self.scaler.transform(data.iloc[:, 10:38].values), dtype=torch.float)
