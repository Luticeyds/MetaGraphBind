import time
import torch
import copy
import argparse

from torch.utils.data import DataLoader
from Dataset.data import PreDataset, FinetuneDataset, ValDataset, Unlabeled_Dataset
from model.Model import Model, TransModel
from trainer.Train import Trainer, MT_Trainer

# Parameter Configuration
parser = argparse.ArgumentParser()

parser.add_argument('--excel_file', type=str, default='data/R_7.18.xlsx', help="Data excel")
parser.add_argument('--train_ratio', type=float, default=0.8, help='train ratio')
parser.add_argument('--batch_size', type=int, default=4096, help='batch size')# R 12792   co 5692
parser.add_argument('--num_features', type=int, default=9, help='number of graph features')
parser.add_argument('--num_classes', type=int, default=34, help='number of GCN output')
parser.add_argument('--num_hidden_GCN', type=int, default=128, help='number of GCN hidden')
parser.add_argument('--num_layers', type=int, default=6, help='number of hidden layers')
parser.add_argument('--num_hidden_MLP1', type=int, default=360, help='number of hidden MLP1')
parser.add_argument('--num_hidden_MLP2', type=int, default=120, help='number of hidden MLP2')
parser.add_argument('--num_output', type=int, default=1, help='number of output')
parser.add_argument('--pooling', type=float, default=0.8, help='pooling rate')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1.51e-04, help='weight decay') # 1.51e-04
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
parser.add_argument('--eps', type=float, default=1e-05, help='eps') # 1e-05
parser.add_argument('--beta1', type=float, default=0.99, help='beta1') # 0.93
parser.add_argument('--beta2', type=float, default=0.999, help='beta2') # 0.993
parser.add_argument('--patience', type=int, default=200, help='patience')
parser.add_argument('--cr_mode', type=str, default='SmoothL1Loss', choices=['MSE', 'SmoothL1Loss'])
parser.add_argument('--lr_mode', type=str, default='' ,choices=['','ReduceLROnPlateau', 'cosineAnnWarm'])
parser.add_argument('--op_mode', type=str, default='AdamW', choices=['Adam', 'SGD'])
parser.add_argument('--model_name', type=str, default='model_2.0', help='model name')
parser.add_argument('--save_name', type=str, default='trans_1')
parser.add_argument('--unlabeled_excel_file', type=str, default='data/mol_combine.12.20.xlsx', help="Data excel")
parser.add_argument('--num_copies', type=int, default='3', help="Number of copies")

def pre_main():
    args = parser.parse_args()
    args.seed = 42
    args.batch_size = 4096
    args.excel_file = 'data/R_7.18.xlsx'
    # model_1
    # args.num_hidden_GCN = 111
    # args.num_hidden_MLP1 = 198
    # args.num_hidden_MLP2 = 145
    args.num_layers = 3
    # model_2
    args.num_hidden_GCN = 144
    args.num_hidden_MLP1 = 197
    args.num_hidden_MLP2 = 149

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.model_name = 'model_pre_1.8'
    start_time = time.time()
    dataset = PreDataset(args)
    # torch.save(dataset, 'data/jess_dataset.pth')
    # dataset = torch.load('data/jess_dataset.pth')
    model = Model(args)
    Train = Trainer(model, dataset, args)
    Train.run()
    print(time.time() - start_time)
    # 清除GPU缓存
    torch.cuda.empty_cache()

def finetune_main():
    args = parser.parse_args()
    args.batch_size = 4096
    args.seed = 42
    args.excel_file = 'data/R_5.22.xlsx'
    args.model_name = 'model_pre_1.8'
    args.learning_rate = 0.001

    # args.num_hidden_GCN = 111
    # args.num_hidden_MLP1 = 198
    # args.num_hidden_MLP2 = 145
    args.num_layers = 3
    #
    args.num_hidden_GCN = 144
    args.num_hidden_MLP1 = 197
    args.num_hidden_MLP2 = 149

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    start_time = time.time()
    dataset1 = FinetuneDataset(args)

    # torch.save(dataset, 'data/finetune_dataset.pth')
    # dataset = torch.load('data/finetune_dataset.pth')
    model1 = Model(args)
    Train1 = Trainer(model1, dataset1, args)
    Train1.fine_tune()
    args.model_name = 'trans_' + args.model_name


    args.excel_file = 'data/new_bro.xlsx'
    dataset2 = FinetuneDataset(args)
    model2 = Model(args)
    Train2 = Trainer(model2, dataset2, args)
    Train2.fine_tune()

    print(time.time() - start_time)
    # 清除GPU缓存
    torch.cuda.empty_cache()

def shap_main():
    args = parser.parse_args()
    args.batch_size = 4096
    args.model_name = 'trans_model_2'
    args.excel_file = 'data/test11.8.xlsx'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    start_time = time.time()
    dataset = ValDataset(args)
    model = Model(args)
    Train = Trainer(model, dataset, args)
    Train.shap_value()
    print(time.time() - start_time)
    # 清除GPU缓存
    torch.cuda.empty_cache()

def mt_main():
    args = parser.parse_args()
    args.device = "cuda:1"
    args.batch_size = 4096
    args.excel_file = 'data/new_bro.xlsx'
    args.save_name = 'model_only_article'
    args.num_copies = 20
    args.epochs = 1000

    # model_1
    args.num_hidden_GCN = 111
    args.num_hidden_MLP1 = 198
    args.num_hidden_MLP2 = 145
    args.num_layers = 3
    # model_2
    # args.num_hidden_GCN = 144
    # args.num_hidden_MLP1 = 197
    # args.num_hidden_MLP2 = 149

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    start_time = time.time()
    # unlabeled_dataset = Unlabeled_Dataset(args)
    # torch.save(unlabeled_dataset, 'data/unlabeled_dataset.pth')
    unlabeled_dataset = torch.load('data/unlabeled_dataset.pth')
    labeled_dataset = FinetuneDataset(args)
    # labeled_dataset = torch.load('data/jess_dataset.pth')
    model = Model(args)
    Train = MT_Trainer(model, labeled_dataset, unlabeled_dataset, args)
    Train.run()
    print(time.time() - start_time)
    # 清除GPU缓存
    torch.cuda.empty_cache()

def mt_ft_main():
    args = parser.parse_args()
    args.device = "cuda:0"
    args.batch_size = 4096
    args.excel_file = 'data/new_bro.xlsx'
    args.unlabeled_excel_file = 'data/gdb_clean_4.16.csv'
    args.model_name = 'model_pre_2_42'  # 'model_2.0'
    args.save_name = 'model_ft_2_5.15'
    args.epochs = 1000
    args.num_copies = 20
    args.seed = 42

    # args.num_hidden_GCN = 111
    # args.num_hidden_MLP1 = 198
    # args.num_hidden_MLP2 = 145
    args.num_layers = 3

    args.num_hidden_GCN = 144
    args.num_hidden_MLP1 = 197
    args.num_hidden_MLP2 = 149

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    start_time = time.time()
    # unlabeled_dataset = Unlabeled_Dataset(args)
    # torch.save(unlabeled_dataset, 'data/unlabeled_dataset.5.17.pth')
    unlabeled_dataset = torch.load('data/unlabeled_dataset.5.17.pth')
    labeled_dataset = FinetuneDataset(args)
    # gcn_state_dict = torch.load('net/' + args.model_name + '.pth')
    state_dict = torch.load('net/' + args.model_name + '.pt')
    model = Model(args)
    # model.gcn1.load_state_dict(gcn_state_dict)
    model.load_state_dict(state_dict)
    Train = MT_Trainer(model, labeled_dataset, unlabeled_dataset, args)
    Train.run()
    print(time.time() - start_time)
    # 清除GPU缓存
    torch.cuda.empty_cache()

def val_main():
    import torch
    args = parser.parse_args()
    # args.device = 'cpu'
    args.batch_size = 4096
    args.excel_file = 'data/article_ligand.xlsx'
    args.model_name = 'model_ft_2_12.27_teacher'
    args.save_name = 'GANdz'

    # args.num_hidden_GCN = 111
    # args.num_hidden_MLP1 = 198
    # args.num_hidden_MLP2 = 145
    args.num_layers = 3
    args.num_hidden_GCN = 144
    args.num_hidden_MLP1 = 197
    args.num_hidden_MLP2 = 149

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    start_time = time.time()
    dataset = ValDataset(args)
    # full_dataset = FinetuneDataset(args)
    val_loader = dataset.val_loader
    state_dict = torch.load('net/' + args.model_name + '.pt')
    print('data ok')
    model = Model(args)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for Graph, features, labels in val_loader:
            Graph, features, labels = Graph.to(args.device), features.to(
                args.device), labels.to(args.device)

            pred = model(Graph, features)
            true = labels  # 真实 logK1

            y_pred.append(pred.cpu())
            y_true.append(true.cpu())
    import pandas as pd
    import torch

    y_pred = torch.cat(y_pred).numpy().flatten()
    y_true = torch.cat(y_true).numpy().flatten()

    df = pd.DataFrame({
        "Experimental_logK1": y_true,
        "Predicted_logK1": y_pred
    })

    df.to_csv("train_parity_data.csv", index=False)

    # Train = Trainer(model, dataset, args)
    # Train.val()
    print(time.time() - start_time)
    # 清除GPU缓存
    torch.cuda.empty_cache()

def wei_to_all():
    args = parser.parse_args()
    args.num_layers = 3

    args.num_hidden_GCN = 144
    args.num_hidden_MLP1 = 197
    args.num_hidden_MLP2 = 149
    args.model_name = 'model_ft_2_2_6.9_teacher'
    state_dict = torch.load('net/' + args.model_name + '.pt')
    model = Model(args)
    model.load_state_dict(state_dict)
    torch.save(model, 'net/' + args.model_name + '_all.pt')

def double_ft():
    model_mid_name = 'model_ft_1_12.27'
    args = parser.parse_args()
    args.device = "cuda:0"
    args.batch_size = 4096
    args.excel_file = 'data/R_5.22.xlsx'
    args.unlabeled_excel_file = 'data/gdb_clean.csv'
    args.model_name = 'model_pre_12.27'  # 'model_2.0'
    args.save_name = model_mid_name
    args.epochs = 1000
    args.num_copies = 0
    args.seed = 42

    # args.num_hidden_GCN = 111
    # args.num_hidden_MLP1 = 198
    # args.num_hidden_MLP2 = 145
    args.num_layers = 3

    args.num_hidden_GCN = 144
    args.num_hidden_MLP1 = 197
    args.num_hidden_MLP2 = 149

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    start_time = time.time()
    # unlabeled_dataset = Unlabeled_Dataset(args)
    # torch.save(unlabeled_dataset, 'data/unlabeled_dataset.6.6.pth')
    unlabeled_dataset = torch.load('data/unlabeled_dataset.6.6.pth')
    labeled_dataset = FinetuneDataset(args)
    # gcn_state_dict = torch.load('net/' + args.model_name + '.pth')
    state_dict = torch.load('net/' + args.model_name + '.pt')
    model = Model(args)
    # model.gcn1.load_state_dict(gcn_state_dict)
    model.load_state_dict(state_dict)
    Train = MT_Trainer(model, labeled_dataset, unlabeled_dataset, args)
    Train.run()
    print('Training time 1:', time.time() - start_time)
    mid_time = time.time()
    # 清除GPU缓存
    torch.cuda.empty_cache()

    args.excel_file = 'data/new_bro.xlsx'
    args.model_name = model_mid_name + '_teacher'
    args.save_name = 'model_ft_2_12.27'
    args.num_copies = 20
    labeled_dataset_2 = FinetuneDataset(args)
    state_dict_2 = torch.load('net/' + args.model_name + '.pt')
    model_2 = Model(args)
    model_2.load_state_dict(state_dict_2)
    Train_2 = MT_Trainer(model_2, labeled_dataset_2, unlabeled_dataset, args)
    Train_2.run()
    print('Training time 2:', time.time() - mid_time)
    print('All time:', time.time() - start_time)
    # 清除GPU缓存
    torch.cuda.empty_cache()

def b_jieduan():
    args = parser.parse_args()
    args.device = "cuda:1"
    args.batch_size = 4096
    args.excel_file = 'data/new_bro.xlsx'
    args.model_name = 'model_only_article'  # 'model_2.0'
    args.save_name = 'model_only_article'
    args.epochs = 1000
    args.num_copies = 0
    args.seed = 42

    # args.num_hidden_GCN = 111
    # args.num_hidden_MLP1 = 198
    # args.num_hidden_MLP2 = 145
    args.num_layers = 3

    args.num_hidden_GCN = 144
    args.num_hidden_MLP1 = 197
    args.num_hidden_MLP2 = 149

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # unlabeled_dataset = Unlabeled_Dataset(args)
    # torch.save(unlabeled_dataset, 'data/unlabeled_dataset.6.6.pth')
    unlabeled_dataset = torch.load('data/unlabeled_dataset.6.6.pth')
    labeled_dataset = FinetuneDataset(args)
    model = Model(args)
    Train = MT_Trainer(model, labeled_dataset, unlabeled_dataset, args)
    Train.run()


if __name__ == '__main__':
    # pre_main()
    # torch.cuda.empty_cache()
    # mt_main()
    # mt_ft_main()
    val_main()
    # wei_to_all()
    # double_ft()

    # b_jieduan()
    # finetune_main()