import argparse
import os
import shutil
import time

import torch

from Dataset.data import PreDataset, FinetuneDataset, ValDataset, Unlabeled_Dataset
from model.Model import Model
from trainer.Train import Trainer, MT_Trainer


def build_parser():
    parser = argparse.ArgumentParser()

    # ========= basic =========
    parser.add_argument('--mode', type=str, default='pipeline', choices=['pipeline', 'pretrain', 'transfer1', 'transfer2', 'val'], help='Run mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str,default='cuda:0' if torch.cuda.is_available() else 'cpu', help='cuda:0 / cpu')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train ratio')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')


    # ========= model =========
    parser.add_argument('--num_features', type=int, default=9, help='Number of graph features')
    parser.add_argument('--num_classes', type=int, default=34, help='Number of GCN output')
    parser.add_argument('--num_hidden_GCN', type=int, default=128, help='GCN hidden dim')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GCN layers')
    parser.add_argument('--num_hidden_MLP1', type=int, default=197, help='MLP hidden dim 1')
    parser.add_argument('--num_hidden_MLP2', type=int, default=149, help='MLP hidden dim 2')
    parser.add_argument('--num_output', type=int, default=1, help='Output dim')
    parser.add_argument('--pooling', type=float, default=0.8, help='Pooling rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--num_copies', type=int, default=20, help="Number of augmented copies for each training sample in finetuning")

    # ========= training =========
    parser.add_argument('--epochs', type=int, default=10, help='Epochs for pretraining')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=1.51e-4, help='Weight decay')
    parser.add_argument('--eps', type=float, default=1e-5, help='AdamW eps')
    parser.add_argument('--beta1', type=float, default=0.99, help='AdamW beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='AdamW beta2')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--cr_mode', type=str, default='SmoothL1Loss', choices=['MSE', 'SmoothL1Loss'])
    parser.add_argument('--lr_mode', type=str, default='', choices=['', 'ReduceLROnPlateau', 'cosineAnnWarm'])
    parser.add_argument('--op_mode', type=str, default='AdamW', choices=['AdamW', 'SGD'])

    # ========= data files =========
    parser.add_argument('--pretrain_excel', type=str, default='data/pretrain_jess_general.xlsx', help='General metal-ligand pretraining dataset')
    parser.add_argument('--transfer1_excel', type=str, default='data/transfer_lnan_subset.xlsx', help='Ln/An transfer dataset')
    parser.add_argument('--transfer2_excel', type=str, default='data/finetune_article_dataset.xlsx', help='Article fine-tuning dataset')
    parser.add_argument('--generated_excel', type=str, default='data/generated_ligand.xlsx', help='Generated dataset in xlsx')

    # ========= scaler files =========
    parser.add_argument('--pretrain_scaler_path', type=str, default='pre_scaler.joblib', help='Scaler path for pretraining')
    parser.add_argument('--transfer1_scaler_path', type=str, default='transfer1_scaler.joblib', help='Scaler path for first transfer')
    parser.add_argument('--transfer2_scaler_path', type=str, default='transfer2_scaler.joblib', help='Scaler path for second transfer')

    # ========= checkpoint names =========
    parser.add_argument('--pretrain_ckpt', type=str, default='model_pretrain', help='Checkpoint name after pretraining')
    parser.add_argument('--transfer1_ckpt', type=str, default='model_transfer1', help='Checkpoint name after first transfer')
    parser.add_argument('--transfer2_ckpt', type=str, default='model_transfer2', help='Checkpoint name after second transfer')
    parser.add_argument('--val_model_name', type=str, default='model_transfer2', help='Checkpoint name used for val')

    # ========= misc =========
    parser.add_argument('--save_name', type=str, default='run', help='Output prefix')

    return parser


def clone_args(args):
    return argparse.Namespace(**vars(args).copy())


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ensure_dirs():
    os.makedirs('net', exist_ok=True)


def copy_ckpt_as(src_name: str, dst_name: str):

    src_path = os.path.join('net', f'net/{src_name}.pt')
    dst_path = os.path.join('net', f'net/{dst_name}.pt')

    if not os.path.exists(src_path):
        raise FileNotFoundError(f'Checkpoint not found: {src_path}')

    if src_path != dst_path:
        shutil.copy2(src_path, dst_path)


def run_pretraining(args):
    stage_args = clone_args(args)
    stage_args.excel_file = args.pretrain_excel
    stage_args.model_name = args.pretrain_ckpt

    print(f'\n[Stage 1/3] Pretraining on: {stage_args.excel_file}')
    print(f'Saving checkpoint as: net/{stage_args.model_name}.pt')

    set_seed(stage_args.seed)
    start = time.time()

    dataset = PreDataset(stage_args)
    model = Model(stage_args)
    trainer = Trainer(model, dataset, stage_args)
    trainer.run()

    print(f'[Stage 1/3] Done. Time: {time.time() - start:.2f}s')
    return stage_args.model_name


def run_transfer(stage_name: str, excel_file: str, load_ckpt: str, save_ckpt: str, args, first=True):

    stage_args = clone_args(args)
    stage_args.excel_file = excel_file
    stage_args.model_name = load_ckpt
    stage_args.save_name = save_ckpt

    print(f'\n[{stage_name}] Fine-tuning on: {stage_args.excel_file}')
    print(f'Loading checkpoint: {load_ckpt}.pt')
    print(f'Final checkpoint will be saved as: {save_ckpt}.pt')

    set_seed(stage_args.seed)
    start = time.time()

    labeled_dataset = FinetuneDataset(stage_args, first)
    unlabeled_dataset = Unlabeled_Dataset(stage_args, first)
    if first:
        state_dict = torch.load(stage_args.model_name + '.pt')
    else:
        state_dict = torch.load(stage_args.model_name + '_student.pt')
    model = Model(stage_args)
    model.load_state_dict(state_dict)
    trainer = MT_Trainer(model, labeled_dataset, unlabeled_dataset, stage_args)
    trainer.run()

    temp_name = f'trans_{load_ckpt}'
    copy_ckpt_as(temp_name, save_ckpt)

    print(f'[{stage_name}] Done. Time: {time.time() - start:.2f}s')
    return save_ckpt


def pipeline_main(args):
    """
    One-click three-step transfer learning:
    1) pretrain on general dataset
    2) transfer on Ln/An dataset
    3) fine-tune on article dataset
    """
    ensure_dirs()
    total_start = time.time()

    ckpt_stage1 = run_pretraining(args)

    ckpt_stage2 = run_transfer(
        stage_name='Stage 2/3',
        excel_file=args.transfer1_excel,
        load_ckpt=ckpt_stage1,
        save_ckpt=args.transfer1_ckpt,
        args=args,
        first=True
    )

    ckpt_stage3 = run_transfer(
        stage_name='Stage 3/3',
        excel_file=args.transfer2_excel,
        load_ckpt=ckpt_stage2,
        save_ckpt=args.transfer2_ckpt,
        args=args,
        first=False
    )

    print('\n===== Pipeline finished =====')
    # print(f'Pretraining checkpoint : net/{ckpt_stage1}.pt')
    print(f'Transfer-1 checkpoint  : net/{ckpt_stage2}.pt')
    print(f'Transfer-2 checkpoint  : net/{ckpt_stage3}.pt')
    print(f'Total time: {time.time() - total_start:.2f}s')


def pretrain_only_main(args):
    ensure_dirs()
    run_pretraining(args)


def transfer1_only_main(args):
    ensure_dirs()
    run_transfer(
        stage_name='Stage 2/3',
        excel_file=args.transfer1_excel,
        load_ckpt=args.pretrain_ckpt,
        save_ckpt=args.transfer1_ckpt,
        args=args
    )


def transfer2_only_main(args):
    ensure_dirs()
    run_transfer(
        stage_name='Stage 3/3',
        excel_file=args.transfer2_excel,
        load_ckpt=args.transfer1_ckpt,
        save_ckpt=args.transfer2_ckpt,
        args=args
    )


def val_main(args):
    ensure_dirs()

    val_args = clone_args(args)
    val_args.excel_file = args.val_excel
    val_args.model_name = args.val_model_name

    print(f'\n[VAL] Dataset: {val_args.excel_file}')
    print(f'[VAL] Model  : net/{val_args.model_name}.pt')

    set_seed(val_args.seed)
    start = time.time()

    dataset = FinetuneDataset(val_args)
    val_loader = dataset.val_loader

    state_dict = torch.load(os.path.join('net', f'{val_args.model_name}.pt'),
                            map_location=val_args.device)

    model = Model(val_args)
    model.load_state_dict(state_dict)
    model.to(val_args.device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for graph, features, labels in val_loader:
            graph = graph.to(val_args.device)
            features = features.to(val_args.device)
            labels = labels.to(val_args.device)

            pred = model(graph, features)
            y_pred.append(pred.cpu())
            y_true.append(labels.cpu())

    y_pred = torch.cat(y_pred).numpy().flatten()
    y_true = torch.cat(y_true).numpy().flatten()

    import pandas as pd
    df = pd.DataFrame({
        "Experimental_logK1": y_true,
        "Predicted_logK1": y_pred
    })
    df.to_csv("train_parity_data.csv", index=False)

    print(f'[VAL] Done. Time: {time.time() - start:.2f}s')
    print('[VAL] Saved to train_parity_data.csv')


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == 'pipeline':
        pipeline_main(args)
    elif args.mode == 'pretrain':
        pretrain_only_main(args)
    elif args.mode == 'transfer1':
        transfer1_only_main(args)
    elif args.mode == 'transfer2':
        transfer2_only_main(args)
    elif args.mode == 'val':
        val_main(args)