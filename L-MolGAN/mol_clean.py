from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from utils import from_smiles
import re
import joblib
import pandas as pd

# ----------------------------- Configuration Parameters -----------------------------
INPUT_FILE = "zinc/3.28.txt"  # Input file name
OUTPUT_FILE = "zinc/clean_3.28.txt"  # Output file name
EXCEL_FILE = "zinc/clean_3.28.xlsx"
RESULT_FILE = "zinc/result_3.28.xlsx"

# -------------------- Define SMARTS patterns for unstable / hard-to-synthesize groups --------------------
unstable_smarts = [
    # Easily hydrolyzed / decomposable groups
    '[OX2]-[OX2]',  # Peroxide
    'S-C(=O)-C',  # Thioester
    '[S](C)(C)O',  # Thioketal / thioacetal
    '[N-]=[N+]=[N-]',  # Azide
    '[N]=[O]',  # Nitroso
    '[N]=[N]',  # Nitrene
    '[N]#[N]',  # Diazo group
    '[NX3][OX1]',  # Nitro-oxygen group
    '[CX3](=O)-[NX3H2]',  # Amide group
    'C=[C]=[C]',  # Conjugated alkene
    'C#C-C#C',  # Consecutive alkynes
    '[C]=[C]-[C]=O',  # α,β-unsaturated carbonyl (Michael acceptor)
    '[CX3](=O)-[OX2H0]',  # Ester group
    '[Cl,Br,I]-C(-C)(-C)-C',  # Unstable benzylic hydrogen (e.g., benzyl chloride)
    '[Cl]-[C;!$(C-[O,N,S])]',  # Organic chlorides (synthetic issues)
    '[PX4]=[OX1]'  # Unstable phosphoryl group
    '[SX2]-[OX2]',  # Sulfur-oxygen (S-O)
    '[SX2]-[SX2]',  # Disulfide bond (S-S)
    '[C]#[N]-[OX2]',  # Cyanide derivatives
    '[NX1]#[CX3]-[CX3]=O',  # Isocyanate
    '[CX3;H0;R0]=[CX3;H0;R0]',  # Highly substituted double bond (no hydrogens, non-ring)

    # High-strain structures
    'C1OC1',  # Epoxide (three-membered ring)
    'C1CC1',  # Three-membered ring (cyclopropane)
    'C1CCC1',  # Four-membered ring (cyclobutane)

    # Hard-to-synthesize structures
    '[r{12-}]',  # Macrocycles (>12-membered rings)
    '[C&$(C@*)][C&$(C@*)]',  # Consecutive chiral centers
]

# Convert to RDKit Mol objects
unstable_patterns = [Chem.MolFromSmarts(s) for s in unstable_smarts]


# ----------------------------- Helper Functions -----------------------------


def is_chelator_valid(mol):
    """Comprehensively check molecular stability and synthetic feasibility."""
    # 1. Skip invalid molecules
    if mol is None:
        return False

    # 2. Sanitize check
    try:
        Chem.SanitizeMol(mol)
    except:
        return False

    # 3. Exclude unstable groups through substructure matching
    for pattern in unstable_patterns:
        if mol.HasSubstructMatch(pattern):
            return False

    return True


def df_to_logk(smi, model_file='trans_1.16_4_all.pt'):
    class ValDataset(Dataset):
        def __init__(self, smi, device='cuda:0'):
            self.scaler = joblib.load('scaler.joblib')
            self.device = device
            self.smiles_list = smi
            self.labels = torch.load('labels.pth', weights_only=False)
            self.feature_df = pd.read_pickle('feature.pkl')
            self.batch_size = 4096
            self.graph_data = [g for g in self.smiles_to_graph() if g.size(0) > 0]
            self.supplement_data = torch.load('supplement_data.pth', weights_only=False)
            self.combined_data = [
                (graph_idx, supp_idx)
                for graph_idx in range(len(self.graph_data))
                for supp_idx in range(len(self.supplement_data))
            ]
            self.co_graph = []
            self.co_su_data = []
            self.co_label = []
            self.data_combine()
            self.data_split()

        def __len__(self):
            return len(self.co_graph)

        def __getitem__(self, idx):
            graph_features = self.co_graph[idx].to(self.device)
            supplement_features = self.co_su_data[idx].to(self.device)
            label = self.co_label[idx].to(self.device)
            return graph_features, supplement_features, label

        def smiles_to_graph(self):
            graph_list = []
            for smiles in self.smiles_list:
                data = from_smiles(smiles[0], with_hydrogen=False, kekulize=True).to(self.device)
                graph_list.append(data)
            return graph_list

        def data_combine(self):
            num = len(self.combined_data)
            for idx in range(num):
                graph_idx, supp_idx = self.combined_data[idx]
                self.co_graph.append(self.graph_data[graph_idx])
                self.co_su_data.append(self.supplement_data[supp_idx])
                self.co_label.append(self.labels[supp_idx])

        def data_split(self):
            self.val_loader = DataLoader(self, batch_size=self.batch_size, shuffle=False)




    device = 'cuda:0'
    try:
        dataloader = ValDataset(smi, device=device).val_loader
    except RuntimeError as e:
        max_value = torch.tensor(0.0, requires_grad=True, device=device)
        with open('output_smi_2.txt', 'a') as f:
            f.write(f"Max Value: {max_value.item()}\n\n")
        print(e)
        return max_value
    model = torch.load(model_file, weights_only=False)
    model.eval()
    model.to(device)

    first_batch = True
    smi_features_logk = []

    with torch.no_grad():
        idx = 0
        for Graph, features, labels in dataloader:
            Graph, features, labels = Graph.to(device), features.to(device), labels.to(device)
            output = model(Graph, features)
            if first_batch:
                y = labels
                out = output
                first_batch = False
            else:
                y = torch.cat((y, labels))
                out = torch.cat((out, output))
                # Now we collect the smi, features, and corresponding det_logk values
            i = 0
            while i < len(labels) / 2:
                smi_idx = smi[idx // 11]  # Get corresponding SMILES string from input list
                Eu_value = output[i] # Extract Eu values
                Am_value = output[i+1]  # Extract Am values
                det_logk = Am_value - Eu_value  # Calculate det_logk
                smi_features_logk.append(
                    (smi_idx, Eu_value.cpu().numpy()[0], Am_value.cpu().numpy()[0], det_logk.cpu().numpy()[0])) # Save SMILES, features, and det_logk
                idx += 1
                i += 1

    # Save results to an Excel file
    smi_df = pd.DataFrame(smi_features_logk, columns=["SMILES", "Eu", "Am", "det_logk"])
    smi_df.to_excel(RESULT_FILE, index=False)

def to_e(smiles):
    feature_data = pd.read_pickle('feature.pkl')

    result_data = []
    feature_data['W'] = feature_data['W'].astype(int)
    for smi in smiles:
        rows_to_add = feature_data[feature_data['W'] != ''].copy()
        for index, row in rows_to_add.iterrows():
            row['SMILES'] = smi
            result_data.append(row)


    result_df = pd.DataFrame(result_data)
    result_df.to_excel(EXCEL_FILE, index=False)

# ----------------------------- Main Workflow -----------------------------
def filter_smiles_file():
    # Read the file and filter out non-SMILES lines
    with open(INPUT_FILE, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith(('Mean', 'Max'))]

    valid_smiles = []
    for line in lines:
        # Extract the SMILES string (ignore possible numeric suffixes)
        smi = re.split(r'\s+', line)[0]
        mol = Chem.MolFromSmiles(smi)
        if is_chelator_valid(mol):
            valid_smiles.append(smi)

    to_e(valid_smiles)

    # Save the results
    with open(OUTPUT_FILE, 'w') as f:
        f.write("\n".join(valid_smiles))

    print(f"Filtering completed! Original number of molecules: {len(lines)}, valid number of molecules: {len(valid_smiles)}")
    print(f"Results have been saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    filter_smiles_file()
