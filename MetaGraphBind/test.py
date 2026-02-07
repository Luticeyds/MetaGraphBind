import main
import torch
from torcheval.metrics.functional import r2_score
from trainer.utils import double_lines_graph, density_graph, save_pooled_graphs_as_png, visualize_molecule_with_shap
from itertools import cycle
from Dataset.data import ValDataset
from model.Model import Model

args = main.parser.parse_args()
args.excel_file = 'data/new_bro.xlsx'
dataset = ValDataset(args)
model = Model(args)
device = args.device
model_state_dict = torch.load('net/model_11.21_all.pt')
model.load_state_dict(model_state_dict)
model.to(device)
loader = dataset.val_loader
model.eval()

model.eval()
first_batch = True
with torch.no_grad():
    for Graph, features, labels in loader:
        Graph, features, labels = Graph.to(device), features.to(
            device), labels.to(device)

        output = model(Graph, features)

        if first_batch:
            y = labels
            out = output
            first_batch = False
        else:
            y = torch.cat((y, labels))
            out = torch.cat((out, output))

    test_r2 = r2_score(out, y)
    print(test_r2)
    density_graph(y, out, test_r2, net_name='all_bro', mode='val', save=False)
