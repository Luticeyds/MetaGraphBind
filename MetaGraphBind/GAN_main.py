from main import parser
from model.MolGAN import GraphGANModel
import torch
from trainer.Train import GANTrainer
from torch_geometric.utils import from_smiles


args = parser.parse_args()
model = GraphGANModel(args)
# 定义生成器优化器
generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=1e-3)

# 初始化 GANTrainer
trainer = GANTrainer(model, generator_optimizer)
trainer.train_generator(epochs=500)
trainer.test_generator(num_samples=1)