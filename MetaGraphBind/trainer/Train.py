import copy
import torch
import random
import numpy as np
import pandas as pd
from captum.attr import IntegratedGradients
from torch.nn import Linear, init, MSELoss, Module
from torch_geometric.nn import GCNConv
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torcheval.metrics.functional import r2_score
from torch_geometric.utils import to_smiles
from rdkit import Chem
from .utils import double_lines_graph, density_graph, save_pooled_graphs_as_png, visualize_molecule_with_shap
from itertools import cycle, islice
class Trainer(object):

    def __init__(self, model, dataset, args):
        """
        Initialize the Trainer class.

        Parameters:
        - model (nn.Module): The neural network model to be trained.
        - dataset (Dataset): The dataset object containing training and testing data loaders.
        - args (Namespace): The arguments containing training parameters.
        """
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.mod = model.to(self.device)
        self.lr = args.lr
        self.patience = args.patience

        # Optimizer
        self.op_mode = args.op_mode
        self.weight_decay = args.weight_decay
        if self.op_mode == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr,betas=(args.beta1, args.beta2),
                                               weight_decay=self.weight_decay, eps=args.eps)
        elif self.op_mode == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=args.momentum,
                                             weight_decay=self.weight_decay)

        # Loss function
        self.cr_mode = args.cr_mode
        if self.cr_mode == 'MSE':
            self.criterion = torch.nn.MSELoss()
        elif self.cr_mode == 'SmoothL1Loss':
            self.criterion = torch.nn.SmoothL1Loss()

        # Learning rate scheduler
        self.lr_mode = args.lr_mode
        if self.lr_mode == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=self.patience//4,
                                               threshold_mode='abs')
        elif self.lr_mode == 'cosineAnnWarm':
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2, eta_min=0)

        self.dataset = dataset
        self.epochs = args.epochs
        self.epoch = 0
        self.model_name = args.model_name
        self.save_name = args.save_name
        self.iteration = 0
        self.train_ls = []
        self.test_ls = []
        self.best_dict = {'train_loss': 100, 'test_loss': 100, 'train_r2': 0.0, 'test_r2': 0.0,
                          'train_out':None, 'test_out':None, 'train_label': None, 'test_label':None,
                          'epoch': 1, 'model_wts': copy.deepcopy(self.model.state_dict()), 'patience': 0,
                          'train_num':None, 'test_num':None, 'train_datas':None, 'test_datas':None}

        self.val_loss = None
        self.val_r2 = None

    def run(self):
        """
        Run the training process. Handles both regular training and k-fold cross-validation.
        """
        for _ in range(self.epochs):
            self.epoch += 1
            self.main()
            if self.best_dict['patience'] >= self.patience:
                break
        self.end()

    def val(self):

        val_out, val_label, _, _ = self.test(val=True)


        pre_values = val_out.detach().cpu().numpy().flatten()
        print('value ok')
        data = {'Pre': pre_values}
        df = pd.DataFrame(data)
        df.to_csv(self.save_name + 'val_out.csv')

    def train(self):
        """
        Train the model for one epoch.

        Returns:
        - out (Tensor): Model outputs.
        - y (Tensor): Ground truth labels.
        - num (Tensor): The sample number.
        - train_loss (float): Average training loss.
        - train_r2 (float): R-squared score for training data.
        """
        self.model.train()
        train_loss = 0.0
        first_batch = True
        loader = self.dataset.train_loader

        for Graph, features, labels in loader:
            Graph, features, labels = Graph.to(self.device), features.to(
                self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            output= self.model(Graph, features)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * len(features)
            if first_batch:
                y = labels
                out = output
                first_batch = False
            else:
                y = torch.cat((y, labels))
                out = torch.cat((out, output))

        train_r2 = r2_score(out, y)
        train_loss /= len(loader.dataset)

        return out, y, train_loss, train_r2

    def test(self, val=False):
        """
        Evaluate the model on the test or validation set.

        Parameters:
        - val (bool): If True, evaluate on the validation set. Otherwise, evaluate on the test set.

        Returns:
        - out (Tensor): Model outputs.
        - y (Tensor): Ground truth labels.
        - num (Tensor): The sample number.
        - test_loss (float): Average test loss.
        - test_r2 (float): R-squared score for test data.
        """
        self.model.eval()
        test_loss = 0.0
        self.model.eval()
        first_batch = True
        if val:
            loader = self.dataset.val_loader
            self.model.load_state_dict(self.best_dict['model_wts'])
        else:
            loader = self.dataset.test_loader

        with torch.no_grad():
            for Graph, features, labels in loader:
                Graph, features, labels = Graph.to(self.device), features.to(
                    self.device), labels.to(self.device)

                output= self.model(Graph, features)
                loss = self.criterion(output, labels)
                test_loss += loss.item() * len(features)

                if first_batch:
                    y = labels
                    out = output
                    first_batch = False
                else:
                    y = torch.cat((y, labels))
                    out = torch.cat((out, output))

            test_r2 = r2_score(out, y)
            test_loss /= len(loader.dataset)

        if val:
            print(f'Val R2: {test_r2:.4f}')
            self.val_loss = test_loss
            self.val_r2 = test_r2
            density_graph(y, out, test_r2, net_name=self.model_name + ' val,best epoch=' + str(self.best_dict['epoch']),
                          mode='val', save=True)

        return out, y, test_loss, test_r2

    def main(self):
        """
        Perform one training and testing iteration, update learning rate, and save the best model parameters.
        """
        train_out, train_label, train_loss, train_r2 = self.train()
        test_out, test_label, test_loss, test_r2 = self.test()

        if self.lr_mode == 'ReduceLROnPlateau':
            self.scheduler.step(test_loss)
        elif self.lr_mode == 'cosineAnnWarm':
            self.scheduler.step(self.epoch)

        self.train_ls.append(train_loss)
        self.test_ls.append(test_loss)

        if test_loss < self.best_dict['test_loss']:
            self.best_dict['train_loss'] = train_loss
            self.best_dict['test_loss'] = test_loss
            self.best_dict['train_r2'] = train_r2
            self.best_dict['test_r2'] = test_r2
            self.best_dict['train_out'] = train_out
            self.best_dict['test_out'] = test_out
            self.best_dict['train_label'] = train_label
            self.best_dict['test_label'] = test_label
            self.best_dict['patience'] = 0
            self.best_dict['epoch'] = self.epoch
            self.best_dict['model_wts'] = copy.deepcopy(self.model.state_dict())
            torch.save(self.best_dict['model_wts'], self.model_name + '.pt')
            print(f'Epoch {self.epoch}, lr: {self.lr:.6f} ,Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f};Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}')
        else:
            self.best_dict['patience'] += 1




    def end(self):
        """
        Handle the final model parameters and results.

        Parameters:
        - val (bool): If True, handle validation results. Otherwise, handle test results.
        """
        print('Best epoch: ', self.best_dict['epoch'])
        double_lines_graph(self.train_ls, self.test_ls, a_label='train loss', b_label='test loss',
                           net_name=self.model_name + 'end')
        density_graph(self.best_dict['train_label'], self.best_dict['train_out'],
                      self.best_dict['train_r2'],
                      net_name=self.model_name + ' train,best epoch=' + str(self.best_dict['epoch']),
                      mode='train', save=True)
        density_graph(self.best_dict['test_label'], self.best_dict['test_out'],
                      self.best_dict['test_r2'],
                      net_name=self.model_name + ' test,best epoch=' + str(self.best_dict['epoch']),
                      mode='test', save=True)
        val_out, val_label,  _, _ = self.test(val=True)

    def fine_tune(self):
        """
        Fine-tune the model, saving the best model based on test R2 score.
        """
        print(self.model_name + ' start')
        model_state_dict = torch.load(self.model_name + '.pt')
        self.model.load_state_dict(model_state_dict)
        self.model = self.model.to(self.device)


        epochs = 10000
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(self.args.beta1, self.args.beta2),
                                           weight_decay=self.args.weight_decay, eps=self.args.eps)
        best_r2 = -10
        patience = 0
        for epoch in range(epochs):
            # Train
            train_out, train_label, train_loss, train_r2 = self.train()
            self.train_ls.append(train_loss)

            # Test
            test_out, test_label, test_loss, test_r2 = self.test()
            self.test_ls.append(test_loss)

            if test_r2 > best_r2:
                patience = 0
                best_r2 = test_r2
                best_train_y = train_label
                best_train_out = train_out
                best_test_y = test_label
                best_test_out = test_out
                model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(model_wts, 'trans_' + self.model_name + '.pt')

            else:
                patience += 1
            if patience >= self.patience:
                break
            print(
                f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}; Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}')
        density_graph(best_train_y, best_train_out, net_name=self.model_name + '_trans_train', mode='train')
        density_graph(best_test_y, best_test_out, net_name=self.model_name + '_trans_test', mode='test')



    def init_normal(self, model):
        """
        Initialize the model parameters.

        For Linear layers, initialize the weights with a normal distribution
        (mean=0, std=0.01) and set biases to zero.

        For GCNConv layers, initialize the internal linear layer's weights with
        a normal distribution (mean=0, std=0.01) and set biases to zero.

        Parameters:
        - model (nn.Module): The model or layer to initialize.
        """
        if isinstance(model, Linear):
            init.normal_(model.weight, mean=0, std=0.01)
            if model.bias is not None:
                init.zeros_(model.bias)
        elif isinstance(model, GCNConv):
            init.normal_(model.lin.weight, mean=0, std=0.01)
            if model.bias is not None:
                init.zeros_(model.bias)

    def shap_value(self):
        print(self.model_name + ' start')

        def distribute_shap_values_to_atoms(atom2clique, num_atoms, shap_values):
            """
            Assign the SHAP value of each clique to the atoms it contains
            :param atom2clique: Tensor or numpy array of shape (2, num_atoms), provides the mapping of each atom to its clique.
            :param num_atoms: Total number of atoms in the molecule.
            :param shap_values: List of SHAP values corresponding to each clique.
            :return: numpy array containing the distributed SHAP values for each atom.
            """
            atom_weights = np.zeros(num_atoms)

            # Traverse each atom and assign a SHAP value based on the clique it belongs to
            for atom_idx in range(atom2clique.shape[1]):
                clique_idx = atom2clique[1, atom_idx]
                if clique_idx < len(shap_values):
                    atom_weights[atom_idx] = shap_values[clique_idx] / (atom2clique[1] == clique_idx).sum().item()

            return atom_weights

        model_state_dict = torch.load(self.model_name + '.pt')
        self.model.load_state_dict(model_state_dict)
        self.model = self.model.to(self.device)

        self.model.eval()
        integrated_gradients = IntegratedGradients(self.model)
        first_batch = True
        shap_value1 = []
        shap_value2 = []
        loader = self.dataset.val_loader
        with torch.no_grad():
            smiles_list = []
            for Graph, features, labels, number in loader:
                Graph, features, labels, number = Graph.to(self.device), features.to(
                    self.device), labels.to(self.device), number.to(self.device)

                output = self.model(Graph, features)

                for target_graph_index in range(len(labels)):

                    sample_graph = self.extract_single_graph(Graph, target_graph_index)
                    smiles_list.append(sample_graph.smiles)
                    num_atoms = Chem.MolFromSmiles(sample_graph.smiles).GetNumAtoms()
                    sample_graph.x.requires_grad = True
                    sample_feature = features[target_graph_index].unsqueeze(0)

                    def forward_func(node_features):
                        graph = copy.deepcopy(sample_graph)
                        graph.x = node_features
                        output= self.model(graph, sample_feature)
                        return output



                    node_weights = torch.ones(sample_graph.x.size(0))

                    integrated_gradients = IntegratedGradients(forward_func)
                    baseline_features = torch.zeros_like(sample_graph.x)

                    node_attributions = integrated_gradients.attribute(
                        inputs=sample_graph.x,
                        baselines=baseline_features
                    )

                    weighted_node_importance = []
                    for node_idx in range(sample_graph.x.size(0)):
                        node_attribution = node_attributions[node_idx]

                        weighted_sum_attribution = torch.sum(node_attribution)
                        weighted_node_importance.append(weighted_sum_attribution.item())
                    shap_value1.append(copy.deepcopy(weighted_node_importance))


                if first_batch:
                    y = labels
                    out = output
                    num = number
                    first_batch = False
                else:
                    y = torch.cat((y, labels))
                    out = torch.cat((out, output))
                    num = torch.cat((num, number))

        pre_list = out.detach().cpu().numpy().reshape(-1)
        labels_list = y.detach().cpu().numpy().reshape(-1)
        num_list = num.detach().cpu().numpy().reshape(-1)

        i = 0
        while i < len(labels_list):
            smiles = smiles_list[i]
            shap1 = shap_value1[i]
            shap2 = shap_value1[i + 1]
            shap3 = shap_value2[i]
            shap4 = shap_value2[i + 1]
            visualize_molecule_with_shap(smiles, shap1, shap2, output_file=f"graph/{i / 2 + 1}_all.png")
            visualize_molecule_with_shap(smiles, shap3, shap4, output_file=f"graph/{i / 2 + 1}_brics.png")
            i += 2

    def extract_single_graph(self, batch, target_graph_index):
        """
        Extract a single graph from a PyTorch Geometric Batch object.

        Args:
        - batch (torch_geometric.data.Batch): A Batch object containing multiple graphs.
        - target_graph_index (int): The index of the target graph to extract.

        Returns:
        - single_graph (torch_geometric.data.Data): The extracted Data object of a single graph.
        """
        # Use the to_data_list method to split the Batch into a list of individual graphs
        data_list = batch.to_data_list()

        # Extract the target graph
        if target_graph_index < len(data_list):
            single_graph = data_list[target_graph_index]
        else:
            raise IndexError(
                f"Target graph index {target_graph_index} is out of range; "
                f"the Batch contains only {len(data_list)} graphs."
            )

        return single_graph


class MT_Trainer(object):
    def __init__(self, model, labeled_dataset, unlabeled_dataset, args):
        """
        Initialize the Trainer class.

        Parameters:
        - model (nn.Module): The neural network model to be trained.
        - dataset (Dataset): The dataset object containing training and testing data loaders.
        - args (Namespace): The arguments containing training parameters.
        """
        self.args = args
        self.device = args.device
        self.student_model = copy.deepcopy(model)
        self.teacher_model = copy.deepcopy(model)
        self.teacher_model.load_state_dict(copy.deepcopy(self.student_model.state_dict()))

        for param in self.teacher_model.parameters():
            param.requires_grad = False


        self.teacher_model.to(self.device)
        self.student_model.to(self.device)

        self.lr = args.lr
        self.patience = args.patience

        # Optimizer
        self.weight_decay = args.weight_decay
        self.optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=self.lr,betas=(args.beta1, args.beta2),
                                           weight_decay=self.weight_decay, eps=args.eps)

        # Loss function
        self.criterion = MSELoss()

        # Learning rate scheduler
        self.lr_mode = args.lr_mode

        self.train_labeled_loader = labeled_dataset.train_loader
        self.unlabeled_loader = unlabeled_dataset.val_loader
        self.test_labeled_loader = labeled_dataset.test_loader
        self.epochs = args.epochs
        self.epoch = 0
        self.global_step = 0
        self.alpha = 0.99
        self.best_T = 0.0
        self.best_S = 0.0
        self.save_name = args.save_name
        self.student_model.freeze_layers()


    def update_ema_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.alpha)
        for ema_param, student_param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
            ema_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)

    def perturb_node_features(self, graph, noise_level = 0.1):
        noise = torch.normal(mean=0, std=noise_level, size=graph.x.size()).to(self.device)
        graph.x += noise
        return graph

    def run(self):
        best_teacher = 0.0
        best_train = 0.0
        patience = 0
        for _ in range(self.epochs):
            self.epoch += 1
            train_loss, train_r2 = self.train()
            if train_r2 > best_train:
                best_train = train_r2

            _, _,teacher_loss, teacher_r2 = self.test(which_model='teacher')
            _, _,student_loss, student_r2 = self.test(which_model='student')
            print(f'Epoch: {self.epoch}, Train Loss: {train_loss:.4f}, Train R2:  {train_r2:.4f}')
            print(f"Teacher Model - Loss: {teacher_loss:.4f}, R2: {teacher_r2:.4f}")
            print(f"Student Model - Loss: {student_loss:.4f}, R2: {student_r2:.4f}")
            if teacher_r2 > best_teacher:
                best_teacher = teacher_r2
                patience = 0
            else:
                patience += 1
            if patience >= self.patience:
                break
        print(f"Best train: {best_train:.4f}")
        print(f"Best teacher: {self.best_T:.4f}")
        print(f"Best student: {self.best_S:.4f}")

        val_out, val_label, avg_loss, R2 = self.test(which_model='val')
        print(f"Best student-Val-Loss: {avg_loss:.4f}, R2: {R2:.4f}")
        pre_list = val_out.detach().cpu().numpy().reshape(-1).reshape(-1, 1)
        labels_list = val_label.detach().cpu().numpy().reshape(-1).reshape(-1, 1)
        pre_values = pre_list.reshape(-1)
        labels_values = labels_list.reshape(-1)
        data = {'Pre': pre_values, 'Labels': labels_values}
        df = pd.DataFrame(data)
        df.to_excel(self.save_name + 'val_out.xlsx')

    def train(self):
        self.student_model.train()
        self.teacher_model.train()
        for (labeled_batch, unlabeled_batch) in zip(self.train_labeled_loader, cycle(self.unlabeled_loader)):
            # Labeled data
            labeled_graph, labeled_features, labeled_labels = labeled_batch
            # Unlabeled data
            unlabeled_graph, unlabeled_features = unlabeled_batch

            # Apply data augmentation to unlabeled data
            student_unlabeled_graph = self.perturb_node_features(unlabeled_graph)
            teacher_unlabeled_graph = unlabeled_graph

            # Forward propagation
            # The student model processes labeled data
            labeled_output = self.student_model(labeled_graph, labeled_features)
            # The student model processes unlabeled data
            student_output_unlabeled = self.student_model(student_unlabeled_graph, unlabeled_features)

            student_output = torch.cat((labeled_output, student_output_unlabeled))

            # The teacher model processes unlabeled data
            with torch.no_grad():
                teacher_output_unlabeled = self.teacher_model(teacher_unlabeled_graph, unlabeled_features)
                teacher_output_labeled = self.teacher_model(labeled_graph, labeled_features)
                teacher_output = torch.cat((teacher_output_labeled, teacher_output_unlabeled))

            # Compute classification loss
            classification_loss = self.criterion(labeled_output, labeled_labels)
            # Compute consistency loss
            consistency_loss = self.criterion(student_output, teacher_output)
            # consistency_loss = self.criterion(student_output_unlabeled, teacher_output_unlabeled)

            # Compute total loss
            consistency_weight = 0.1  # Weight of the consistency loss
            total_loss = classification_loss + consistency_weight * consistency_loss

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Update the teacher model parameters via EMA
            self.global_step += 1
            self.update_ema_variables()

            label_r2 = r2_score(labeled_output, labeled_labels)

        return total_loss.item(), label_r2

    def test(self, which_model='teacher'):
        if which_model == 'teacher':
            model = self.teacher_model
        elif which_model == 'student':
            model = self.student_model
        elif which_model == 'val':
            model = self.student_model
            state_dict = torch.load(self.save_name + '.pt')
            model.load_state_dict(state_dict)
        model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        first_batch = True
        total = 0

        with torch.no_grad():  # No gradient is needed in the testing phase
            for data in self.test_labeled_loader:
                graph, features, labels = data
                graph, features, labels = graph.to(self.device), features.to(self.device), labels.to(self.device)

                # forward propagation
                outputs = model(graph, features)
                loss = self.criterion(outputs, labels)

                # calculate the total loss
                total_loss += loss.item()

                # Calculate the number of samples with correct predictions
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)

                if first_batch:
                    y = labels
                    out = outputs
                    first_batch = False
                else:
                    y = torch.cat((y, labels))
                    out = torch.cat((out, outputs))

        # Calculate loss and accuracy
        avg_loss = total_loss / len(self.test_labeled_loader)
        R2 = r2_score(out, y)
        best_test = self.best_T if which_model == 'teacher' else self.best_S
        if R2 > best_test:
            density_graph( y, out, R2,
                          net_name=self.save_name + which_model,
                          mode='test', save=False)

            if which_model == 'teacher':
                torch.save(copy.deepcopy(self.teacher_model.state_dict()), self.save_name + '_teacher.pt')
                self.best_T = R2
                print(f"better teacher:{self.best_T}")
            else:
                torch.save(copy.deepcopy(self.student_model.state_dict()), self.save_name + '_student.pt')
                self.best_S = R2
                print(f"better student:{self.best_S}")

        return out, y,  avg_loss, R2




class GANTrainer:
    """
    Utility class for training and testing the GraphGANModel.

    Attributes:
        model: The GraphGANModel.
        generator_optimizer: The optimizer used for the generator.
    """
    def __init__(self, model, generator_optimizer, features=None):
        """
        Initialize the GANTrainer.

        Args:
            model: The GraphGANModel.
            generator_optimizer: The optimizer used to optimize the generator.
            features: Optional supplementary features for the reward network.
        """
        self.model = model
        self.generator_optimizer = generator_optimizer
        self.features = features  # Supplementary features loaded from Excel

    def train_generator(self, epochs=10):
        """
        Train the generator, focusing only on the interaction between
        the generator and the reward network.

        Args:
            epochs: Number of training epochs.
        """
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0

            # 1. Generate latent variables
            embeddings = self.model.sample_z(batch_dim=1)  # Generate one molecular embedding at a time

            # 2. Generate molecular graphs through the generator
            outputs_data, value = self.model(embeddings)  # The generator outputs molecular graphs

            # 3. Determine the reward from the model feedback value
            reward = value

            # 4. Generator loss: maximize the reward value
            generator_loss = -reward  # The negative sign indicates reward maximization

            # 5. Optimize the generator
            self.generator_optimizer.zero_grad()
            generator_loss.backward()
            self.generator_optimizer.step()

            total_loss += generator_loss.item()

            # Print logs
            print(f"Epoch {epoch + 1}/{epochs}: Generator Loss = {total_loss:.4f}")

    def test_generator(self, num_samples=10):
        """
        Test the generator by evaluating the validity of generated molecules
        and their reward values.

        Args:
            num_samples: Number of generated molecular samples for testing.
        """
        self.model.eval()
        valid_count = 0
        rewards = []

        with torch.no_grad():
            for _ in range(num_samples):
                # 1. Generate latent variables
                embeddings = self.model.sample_z(batch_dim=1)

                # 2. Generate molecular graphs
                outputs_data, value = self.model(embeddings)

                # 3. Evaluate the reward
                reward = value
                rewards.append(reward.item())

                # 4. Check validity
                try:
                    smiles = to_smiles(outputs_data)
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        valid_count += 1
                except:
                    pass

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        print(f"Valid Molecules: {valid_count}/{num_samples}, Average Reward: {avg_reward:.4f}")






