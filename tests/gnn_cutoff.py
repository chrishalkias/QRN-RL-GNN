import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from base.model import GNN
from base.repeaters import RepeaterNetwork
from labellines import *

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "figure.autolayout" : True,
})

def plot_gnn_cutoff(model_path, 
                             n_nodes=4, 
                             cutoff_tau_ratio = 1,
                             savefig = True):
    """
    Generates instructive plots about the GNN's inner logic and latent representations.
    
    1. Decision Phase Diagram: Heatmap of Q(Swap)-Q(Entangle) vs Link Fidelities.
    2. Latent Space PCA: 2D projection of node embeddings colored by policy decision.
    """

    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir =  model_path.rsplit('/', 1)[0] + '/'
    
    # Load Model
    model = GNN(node_dim=2, output_dim=2).to(device)
    if isinstance(model_path, str):
        weights = torch.load(model_path, map_location=device)
    else:
        weights = model_path
    model.load_state_dict(weights)
    model.eval()

    print(f"--- Analyzing GNN Mechanisms (N={n_nodes}) ---")

    # =========================================================================
    #  ======= PLOT 1: Decision Phase Diagram (Sensitivity Analysis) ==========
    # =========================================================================
    # Hypothesis: The agent should only SWAP if both Left and Right fidelities 
    # are above a certain threshold. We will map this threshold.
    
    print("Generating Decision Phase Diagram...")
    resolution = 20
    fidelities = np.linspace(0.01, 1.0, resolution)
    
    # We focus on Node 1 (indexes 0, [1], 2, 3)
    target_node = 1 
    sensitivity_grid = np.zeros((resolution, resolution))
    
    # Grid Search over Left (0-1) and Right (1-2) Link Fidelities
    for i, f_left in enumerate(fidelities):
        for j, f_right in enumerate(fidelities):
            # Create a clean environment
            env = RepeaterNetwork(n=n_nodes)
            
            # Manually set the specific link qualities we want to test
            env.setLink((0, 1), f_left, linkType=1)
            env.setLink((1, env.n -1), f_right, linkType=1)
            
            # Get state and infer
            data = env.tensorState().to(device)
            if data.edge_attr is not None and data.edge_attr.dim() == 1:
                data.edge_attr = data.edge_attr.view(-1, 1)
                
            with torch.no_grad():
                q_values = model(data).cpu().numpy()
            
            # Calculate Preference: Q(Swap) - Q(Entangle)
            # > 0 means Swap, < 0 means Entangle
            pref = q_values[target_node, 1] - q_values[target_node, 0]
            sensitivity_grid[i, j] = pref

    # Plotting Phase Diagram
    plt.figure(figsize=(10, 8))
    # Origin='lower' puts (0,0) at bottom-left
    ax = sns.heatmap(sensitivity_grid, xticklabels=np.round(fidelities, 2), 
                     yticklabels=np.round(fidelities, 2), cmap="RdYlGn", center=0)
    
    plt.locator_params(axis='both', nbins=10)
    
    ax.invert_yaxis() # Ensure 0 is at bottom
    expr = r'$Q_{swap}-Q_{entangle}$'
    plt.title(f"Decision Phase Diagram ({expr}) \nDoes the agent want to Swap?", fontsize=18)
    plt.xlabel(r"Fidelity of Right Link ($F_2$)", fontsize=16)
    plt.ylabel(f"Fidelity of Left Link ($F_1$)", fontsize=16)

    # --- ADDED: Theoretical Boundary Line (F1 * F2 = e^-0.5) ---
    threshold_val = np.exp(-cutoff_tau_ratio)
    
    # 1. Generate X coordinates in "Heatmap Index Space" (0 to resolution-1)
    # We start slightly above 0 to avoid division by zero errors
    x_indices = np.linspace(0.01, resolution , 200)
    
    # 2. Convert indices to physical values (0.0 to 1.0)
    x_phys = x_indices / (resolution - 1)
    
    # 3. Calculate physical Y using the equation: y = threshold / x
    y_phys = threshold_val / x_phys
    
    # 4. Convert physical Y back to "Heatmap Index Space"
    y_indices = y_phys * (resolution - 1)
    
    # 5. Plot the line
    # We restrict ylim to keep the plot neat (ignoring the asymptote)
    plt.plot(x_indices, y_indices, color='cyan', linestyle='--', linewidth=4, label=r'$F_1 \cdot F_2 = e^{-t_{c}/\tau}$')

    plt.legend(loc='upper left', frameon=True, fontsize=16)
    # plt.ylim(0, resolution - 1)
    
    # Add a contour line for the Decision Boundary (where Preference = 0)
    # We need to map grid indices to plot coordinates
    cnt = plt.contour(np.arange(0.5, resolution), np.arange(0.5, resolution), 
                      sensitivity_grid, levels=[0], colors='black', linewidths=2, linestyles='--')
    plt.clabel(cnt, inline=True, fontsize=11, fmt='Decision Boundary')

    # plt.locator_params(axis='both', nbins=4)

    plt.tight_layout()
    plt.savefig(save_dir +"phase_diagram.png") if savefig else None
    plt.show()
    plt.close()

    # =========================================================================
    #  ======= PLOT 2: Preference vs Individual Link Fidelities ===============
    # =========================================================================
    print("Generating Preference vs Individual Fidelities Plot...")
    
    # We use the sensitivity_grid already calculated in Plot 1.
    # sensitivity_grid[i, j] corresponds to f_left (i) and f_right (j)
    
    # Choose fixed values for the complementary link
    # E.g., fix right link at a high fidelity (0.9) to see left link's impact
    fixed_fid = 1
    fixed_high_idx = int(fixed_fid * (resolution - 1))
    
    # E.g., fix left link at a low fidelity (0.3) to see right link's impact
    fixed_low_idx = int(fixed_fid * (resolution - 1))
    
    # Extract 1D slices from the grid
    # Solid line: Vary Left (F1), Fix Right (F2) at 0.9
    vary_left_pref = sensitivity_grid[:, fixed_high_idx]
    
    # Dashed line: Vary Right (F2), Fix Left (F1) at 0.3
    vary_right_pref = sensitivity_grid[fixed_low_idx, :]
    
    plt.figure(figsize=(8, 6))
    
    plt.plot(fidelities, vary_left_pref, color='black', linestyle='-', linewidth=2, 
             label=r'$F_L$, (fixed $F_R=1$)')
             
    plt.plot(fidelities, vary_right_pref, color='black', linestyle='--', linewidth=2, 
             label=r'$F_R$, (fixed $F_L=1$)')
             
    plt.axhline(0, color='red', linestyle=':', alpha=0.7)
    plt.axvline(x=np.exp(-cutoff_tau_ratio), color='lightblue', linestyle='--', linewidth=2.5, label=r'Cutoff line $F=e^{-t_c/\tau}$')
    plt.xlabel('Link Fidelity', fontsize=16)
    plt.ylabel(r'Swap Preference ($Q_{swap} - Q_{entangle}$)', fontsize=16)
    plt.title('Agent Preference vs. Individual Link Fidelities', fontsize=18)
    
    plt.legend(loc='best', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(save_dir + "preference_1d.png") if savefig else None
    plt.show()
    plt.close()

    



if __name__ == "__main__":
    # Example Usage
    path = 'assets/trained_models/d(14-2)l4u6e7000m60p70a98t150c30/d(14-2)l4u6e7000m60p70a98t150c30.pth'
    plot_gnn_cutoff(model_path=path, 
                             n_nodes=5, 
                             cutoff_tau_ratio=0.53,
                             savefig=True)