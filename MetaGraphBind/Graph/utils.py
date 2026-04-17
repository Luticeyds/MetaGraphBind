import torch
import numpy as np
from matplotlib import cm
from matplotlib import ticker
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import seaborn as sns
import os

def double_lines_graph(a, b, a_label='', b_label='', net_name = 'net'):
    """
    Plot a graph with two lines.

    Parameters:
    - a (array-like): Data for the first line.
    - b (array-like): Data for the second line.
    - a_label (str, optional): Label for the first line. Defaults to ''.
    - b_label (str, optional): Label for the second line. Defaults to ''.
    - net_name (str, optional): Name of the network for saving the plot. Defaults to 'net'.
    """
    plt.plot(a, c='red', label=a_label)
    plt.plot(b, c='blue', label=b_label)
    plt.legend()
    plt.savefig(net_name + '.png', dpi=1000, bbox_inches="tight")
    # plt.show() # Uncomment to display the plot
    plt.close()

def r2_graph(true_values, predicted_values, r2, point_size=10, net_name = 'net'):
    """
    Plot an R2 regression graph.

    Parameters:
    - true_values (array-like or Tensor): True values.
    - predicted_values (array-like or Tensor): Predicted values.
    - r2 (float): R2 score.
    - point_size (int, optional): Size of the points in the scatter plot. Defaults to 10.
    - net_name (str, optional): Name of the network for saving the plot. Defaults to 'net'.
    """
    if type(true_values) == torch.Tensor:
        true_values = true_values.detach().cpu().numpy()
    if type(predicted_values) == torch.Tensor:
        predicted_values = predicted_values.cpu().detach().numpy()
    plt.scatter(true_values, predicted_values, color='blue', s=point_size)
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='black')
    plt.title('$R^2 = {:.2f}$'.format(r2))
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.savefig(net_name + '.png')
    # plt.show() # Uncomment to display the plot
    plt.close()

def density_graph(true_values, predicted_values, r2 = 0.0, net_name = 'net', mode = 'train', save = False):
    """
    Plot an R2 density regression graph.

    Parameters:
    - true_values (array-like or Tensor): True values.
    - predicted_values (array-like or Tensor): Predicted values.
    - r2 (float, optional): R2 score. Defaults to 0.0.
    - net_name (str, optional): Name of the network for saving the plot. Defaults to 'net'.
    - mode (str, optional): Mode of the plot (e.g., 'train', 'test'). Defaults to 'train'.
    - save (bool, optional): Whether to save evaluation indicators. Defaults to False.
    """
    # Convert Tensor to numpy array if needed
    if type(true_values) == torch.Tensor:
        true_values = true_values.detach().cpu().numpy()
    if type(predicted_values) == torch.Tensor:
        predicted_values = predicted_values.cpu().detach().numpy()

    x = np.squeeze(true_values)
    y = np.squeeze(predicted_values)

    if save:
        # Calculation of evaluation indicators
        r2 = r2_score(x, y)
        mae = mean_absolute_error(x, y)
        mse = mean_squared_error(x, y)
        rmse = np.sqrt(mse)

    # Kernel density estimation
    x_and_y = np.vstack([x, y])
    try:
        kde = gaussian_kde(x_and_y)
        z = kde(x_and_y)
    except:
        pca = PCA(n_components=2)
        data_fit = pca.fit_transform(x_and_y)
        kde = gaussian_kde(data_fit.T)
        z = kde(x_and_y)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    is_cbar = True

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4), dpi=1000)
    # cmap: bwr、Spectral_r、viridis_r、spring、gist_rainbow_r、RdBu_r
    # Usable colormaps - https://matplotlib.org/tutorials/colors/colormaps.html
    my_cmap = "RdYlBu_r"
    ax.scatter(x, y, c=z, cmap=my_cmap, s=5)
    plt.plot([min(x), max(x)], [min(x), max(x)], color='black')

    # Customize tick labels
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_color('black') for label in labels]
    plt.tick_params(axis='x', direction='out', labelsize=13, length=4.6, width=1.15)
    plt.tick_params(axis='y', direction='out', labelsize=13, length=4.6, width=1.15)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Add colorbar
    if is_cbar:
        norm = colors.Normalize(vmin=np.min(z), vmax=np.max(z))
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=my_cmap), ax=ax)
        cbar.ax.set_ylabel("Scatter Density", labelpad=12)
        cbar.ax.tick_params(labelsize=12)
        labels = cbar.ax.get_xticklabels() + cbar.ax.get_yticklabels()
        [label.set_color('black') for label in labels]
        tick_locator = ticker.MaxNLocator(nbins=8)
        cbar.locator = tick_locator
        cbar.update_ticks()

    # Bold the spines
    lw = 1.25
    ax.spines["right"].set_linewidth(lw)
    ax.spines["left"].set_linewidth(lw)
    ax.spines["top"].set_linewidth(lw)
    ax.spines["bottom"].set_linewidth(lw)

    # Set axis labels and title
    plt.xlabel("True Values", labelpad=8)
    plt.ylabel("Predicted Values", labelpad=8)

    if save:
        plt.title('{} '.format(mode), pad=12)
        # Display evaluation indicators
        textstr = '\n'.join((
            f'$R^2 = {r2:.4f}$',
            f'MAE: {mae:.4f}',
            f'MSE: {mse:.4f}',
            f'RMSE: {rmse:.4f}'
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,verticalalignment='top', bbox=props)
    else:
        plt.title('{}, $R^2 = {:.4f}$ '.format(mode, r2), pad=12)

    # Display grid
    plt.grid(alpha=0.360, ls="--", which="major", color="#A9A9A9")
    plt.tight_layout()
    plt.savefig(net_name + '.png', dpi=1000, bbox_inches="tight")
    # plt.show() # Uncomment to display the plot
    plt.close()

def save_pooled_graphs_as_png(smiles_list, prefix, num, molsPerRow=1):
    """
    Save a list of SMILES strings as PNG files.

    Parameters:
    - smiles_list (list): List of SMILES strings.
    - prefix (str): Prefix for the saved PNG file names.
    - num (int): Index number for the current set of SMILES.
    - molsPerRow (int): Number of molecules per row in the grid image.
    """

    # Ensure the target directory exists; if not, create it
    os.makedirs(f'./{prefix}', exist_ok=True)

    # Convert SMILES strings to RDKit molecule objects
    mol = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    # Create a grid image of the molecules
    img = Draw.MolsToGridImage(mol, molsPerRow=molsPerRow)

    # Display and save the image
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(f'./{prefix}/{num}.png', dpi=1000)
    plt.close()


def save_mols_as_png(mols_list, prefix, num, molsPerRow=1):

    # Ensure the target directory exists; if not, create it
    os.makedirs(f'./{prefix}', exist_ok=True)
    # Create a grid image of the molecules
    img = Draw.MolsToGridImage(mols_list, molsPerRow=molsPerRow)
    # Display and save the image
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(f'./{prefix}/{num}.png', dpi=1000)
    plt.close()

def save_fp_contribution(data, filename):
    # Calculate the average contribution value for each functional group
    grouped_means = data.groupby('fg_name')['contribution'].mean()

    # Add a column to distinguish the sign of contribution values
    data['sign'] = data['contribution'].apply(lambda x: 'positive' if x >= 0 else 'negative')

    # Create the figure
    plt.figure(figsize=(14, 18))
    sns.set(style='white', font_scale=2)

    # Draw a split violin plot
    sns.violinplot(
        x='contribution',
        y='fg_name',
        data=data,
        inner=None,
        hue='sign',
        split=True,
        cut=0,
        palette={'negative': 'blue', 'positive': 'orange'}
    )

    # Draw mean lines and labels
    ax = plt.gca()
    for i, (group, mean) in enumerate(grouped_means.items()):
        ax.axvline(
            x=mean,
            ymin=i / len(grouped_means),
            ymax=(i + 1) / len(grouped_means),
            color='green',
            linestyle='--'
        )

    # Move the mean labels to the left side of the functional group labels, close to the y-axis
    for i, (group, mean) in enumerate(grouped_means.items()):
        ax.text(
            ax.get_xlim()[0] + 8,
            i,
            f'{mean:.2f}',
            fontsize=20,
            color='black',
            va='center',
            ha='right'
        )

    # Add the average attribution label at the top
    ax.text(
        ax.get_xlim()[0] + 10,
        -1,
        'Average  \nAttribution',
        fontsize=20,
        color='black',
        va='center',
        ha='right'
    )

    # Add labels and title
    plt.xlabel('Contribution', fontsize=20)
    plt.ylabel('Functional Group', fontsize=20)
    plt.savefig(filename, dpi=1500)
    plt.close()


def visualize_molecule_with_attention(molecule_smiles, attention_scores_list, output_file="molecule_attention.png",
                                      colormap="Greens", title=['A', 'B1', 'B2', 'C1', 'C2', 'D']):
    """
    Visualize a molecule with multiple sets of attention scores.

    Parameters:
    - molecule_smiles (str): SMILES string of the molecule to be visualized.
    - attention_scores_list (list of lists or np.arrays): List containing multiple sets of attention scores for each atom in the molecule.
    - output_file (str): File path to save the output image.
    - colormap (str): Name of the colormap to use for coloring the atoms based on attention scores.
    """
    # Create molecule from SMILES string
    mol = Chem.MolFromSmiles(molecule_smiles)

    # Generate 2D coordinates for the molecule (used for visualization)
    rdDepictor.Compute2DCoords(mol)

    # Set up the figure for multiple subplots (3 rows, 2 columns)
    num_scores = len(attention_scores_list)
    fig, axes = plt.subplots(3, 2, figsize=(12, 15), gridspec_kw={'wspace': 0.05, 'hspace': 0.1, 'left': 0.01,
                                                                  'right': 0.9, 'top': 0.95, 'bottom': 0.05})
    axes = axes.flatten()

    cmap = plt.get_cmap(colormap)
    first = True
    for i, attention_scores in enumerate(attention_scores_list):
        # Convert attention scores to numpy array
        try:
            attention_scores = np.array(attention_scores.detach().cpu())
        except:
            attention_scores = np.array(attention_scores)
        # Normalize attention scores
        if first:
            non_zero_attention_scores = attention_scores[attention_scores != 0]
            base_min, base_max = np.min(non_zero_attention_scores), np.max(non_zero_attention_scores)
            denominator = base_max - base_min
            first = False
        else:
            first = True
        normalized_scores = (attention_scores - base_min) / denominator

        # Set up colors for each atom based on attention scores using the specified colormap
        colors = []
        for idx, score in enumerate(attention_scores):
            if score == 0:
                colors.append((0.7, 0.7, 0.7))  # Grey color for score=0 to indicate hidden atoms
            else:
                colors.append(mcolors.to_rgb(cmap(normalized_scores[idx])))

        # Create a drawer to draw the molecule with highlighted atoms
        drawer = rdMolDraw2D.MolDraw2DCairo(800, 800)  # Set image size to 500x500 pixels
        draw_opts = drawer.drawOptions()
        draw_opts.useBWAtomPalette()  # Use black & white palette for atoms not highlighted

        # Prepare a dictionary to highlight atoms with their corresponding colors
        highlight_dict = {idx: colors[idx] for idx in range(mol.GetNumAtoms())}

        # Highlight edges based on atom scores
        # Highlight edges based on atom scores
        highlight_bonds = {}
        draw_opts.bondLineWidth = 1.5  # Set a thicker bond width for highlighted bonds
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if attention_scores[start] == 0 or attention_scores[end] == 0:
                # If either atom has score=0, use grey
                highlight_bonds[bond.GetIdx()] = (0.7, 0.7, 0.7)
            else:
                # Otherwise, use the average color of the two atoms
                avg_color = ((np.array(colors[start]) + np.array(colors[end])) / 2).tolist()
                highlight_bonds[bond.GetIdx()] = tuple(avg_color)

        # Draw the molecule with attention-based highlighting
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=list(highlight_dict.keys()),
                                           highlightAtomColors=highlight_dict,
                                           highlightBonds=list(highlight_bonds.keys()),
                                           highlightBondColors=highlight_bonds)
        drawer.FinishDrawing()

        # Convert drawing to a PNG image
        png = drawer.GetDrawingText()

        # Save the image to a temporary file
        temp_output_file = f"temp_molecule_attention_{i}.png"
        with open(temp_output_file, "wb") as f:
            f.write(png)

        # Display the image using matplotlib
        img = plt.imread(temp_output_file)
        axes[i].imshow(img, aspect='equal')
        axes[i].axis('off')  # Hide axes for better visualization
        axes[i].set_title(title[i], fontsize=20)


    # Hide any unused subplots
    for j in range(num_scores, len(axes)):
        axes[j].axis('off')

    # Add a single colorbar to indicate the attention score range
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.03, pad=0.05, aspect=35).ax.tick_params(labelsize=20)

    plt.tight_layout()
    plt.savefig(output_file, dpi=1000, bbox_inches='tight')
    plt.close()


def visualize_molecule_with_shap(molecule_smiles, shap_values1, shap_values2, output_file="molecule_combined_shap.png", colormap="bwr"):
    """
    Visualize a molecule with three sets of SHAP values in a single image.

    Parameters:
    - molecule_smiles (str): SMILES string of the molecule to be visualized.
    - shap_values1 (list of floats): SHAP values for the first atom set (e.g., Eu).
    - shap_values2 (list of floats): SHAP values for the second atom set (e.g., Am).
    - output_file (str): File path to save the output image.
    - colormap (str): Name of the colormap to use for coloring the atoms based on SHAP values.
    """

    # Create molecule from SMILES string
    mol = Chem.MolFromSmiles(molecule_smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")

    # Generate 2D coordinates for the molecule (used for visualization)
    rdDepictor.Compute2DCoords(mol)
    shap_values_diff = list(map(lambda x: x[0] - x[1], zip(shap_values2, shap_values1)))
    # Set up the figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    shap_values_list = [shap_values1, shap_values2, shap_values_diff]
    titles = ['Eu', 'Am', 'Diff']

    for i, (ax, shap_values, title) in enumerate(zip(axes, shap_values_list, titles)):
        # Convert SHAP values to numpy array
        shap_values = np.array(shap_values)

        # Normalize SHAP values for coloring to be in the range [-1, 1]
        max_abs_value = np.max(np.abs(shap_values))
        if max_abs_value > 0:
            # Normalize to [-1, 1]
            normalized_scores = shap_values / max_abs_value
        else:
            normalized_scores = np.zeros_like(shap_values)  # If all SHAP values are zero

        # Set up colors for each atom based on SHAP values using the specified colormap
        cmap = plt.get_cmap(colormap)
        colors = [mcolors.to_rgb(cmap((score + 1) / 2)) for score in normalized_scores]

        # Create a drawer to draw the molecule with highlighted atoms
        drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)  # Set image size to 500x500 pixels
        draw_opts = drawer.drawOptions()
        draw_opts.useBWAtomPalette()  # Use black & white palette for atoms not highlighted

        # Prepare a dictionary to highlight atoms with their corresponding colors
        highlight_dict = {idx: colors[idx] for idx in range(mol.GetNumAtoms())}

        # Draw the molecule with SHAP-based highlighting
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=list(highlight_dict.keys()),
                                           highlightAtomColors=highlight_dict)
        drawer.FinishDrawing()

        # Convert drawing to a PNG image
        png = drawer.GetDrawingText()

        # Save the intermediate image
        temp_file = f"temp_molecule_{i}.png"
        with open(temp_file, "wb") as f:
            f.write(png)

        # Display the image using matplotlib
        img = plt.imread(temp_file)
        ax.imshow(img, aspect='equal')
        ax.axis('off')  # Hide axes for better visualization
        ax.set_title(title, fontsize=16, fontweight='bold')

    # Add a common colorbar to indicate the SHAP value range
    norm = plt.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.1)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


# Example usage:
# visualize_molecule_with_shap("CCO", [0.02, -0.001, 0.003], output_file="molecule_shap.png")
if __name__ == "__main__":
    # molecule_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    # attention_scores_list = [
    #     [0.1, 0.3, 0.8, 0.5, 0.2, 0.9, 0.7, 0.4, 0.6, 0.3, 0.5, 0.1, 0.2, 0.4],
    #     [0, 0.3, 0.8, 0.5, 0, 0.9, 0.7, 0.4, 0.6, 0.3, 0.5, 0, 0.2, 0.4],
    #     [0, 0.1, 0.6, 0.8, 0, 0.5, 0.7, 0.2, 0.3, 0.5, 0.4, 0, 0.2, 0.1],
    #     [0, 0, 0.6, 0.8, 0, 0.5, 0.7, 0, 0.3, 0.5, 0.4, 0, 0.2, 0],
    #     [0, 0, 0.3, 0.7, 0, 0.6, 0.8, 0, 0.5, 0.3, 0.7, 0, 0.4, 0],
    #     [0, 0, 0, 0.7, 0, 0.6, 0.8, 0, 0.5, 0, 0.7, 0, 0.4, 0]
    # ]
    # visualize_molecule_with_attention(molecule_smiles, attention_scores_list, colormap='RdBu_r')
    visualize_molecule_with_shap('O=C([O-])C(CO)C1=CC=C2C=CC3=CC=C(C(C(=O)[O-])C(=O)[O-])N=C3C2=N1',
                                 [0.0043062857084237086, 0.0076679454075505565, -0.042491643713367944, 0.025407631652507243, 0.0018936465491465552, -0.4457184281702958, 0.029145015102945544, -0.003567590203653507, -0.003567590203653507, 0.029145015102945544, -0.003567590203653507, -0.003567590203653507, 0.029145015102945544, -0.003567590203653507, -0.003567590203653507, 0.029145015102945544, 0.025407631652507243, 0.014472386669205323, 0.001499642822573529, 0.012088468444194356, 0.014472386669205323, 0.001499642822573529, 0.012088468444194356, 0.010352623441564492, 0.029145015102945544, 0.029145015102945544, 0.010352623441564492]
                                 , output_file="molecule_shap.png")