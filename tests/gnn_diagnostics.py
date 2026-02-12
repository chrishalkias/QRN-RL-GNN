import torch
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.pyplot as plt
import pandas as pd
from base.model import GNN
from base.repeaters import RepeaterNetwork
import matplotlib.colors as mcolors
import numpy as np
import os

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import GATv2Conv

# Placeholder for the class if not available in the context (assuming generic structure based on prompt)
# You likely already have the GNN class definition in your script.

def visualize_gnn_diagnostics(model_path, n_nodes=4, cutoff_tau_ratio=1, savefig=True):
    """
    Visualizes GNN weights (Node features) and Continuous Edge Feature analysis.
    
    Args:
        model_path (str or dict): Path to .pth file or state_dict object.
        n_nodes (int): Number of nodes for the diagnostic states.
    """
    # --- 1. Load Model ---
    # Handle path parsing safely
    if isinstance(model_path, str):
        save_img_dir = model_path.rsplit('/', 1)[0] + '/'
        weights = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        save_img_dir = "./"
        weights = model_path

    # Initialize model (Assuming strict architecture match to weights)
    # We use CPU for visualization to avoid CUDA overhead for simple plots
    device = torch.device('cpu') 
    
    # Note: Ensure your GNN class definition matches the saved weights
    # Assuming GNN(node_dim=2, output_dim=2) based on your snippet
    try:
        model = GNN(node_dim=2, output_dim=2).to(device)
        model.load_state_dict(weights)
    except NameError:
        print("Warning: 'GNN' class not found in local scope. Ensure the class definition is available.")
        return

    model.eval()

    # --- 2. Prepare Data for Plotting ---
    print("--- Visualizing Learned Weights & Edge Propagation ---")
    
    # Create a 2x3 Figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # --- ROW 1: NODE WEIGHTS (Discrete/Binary) ---
    
    # 1.1 Encoder: GATv2 Projection (Source Nodes)
    gat_layer = model.encoder[0]
    
    # robust attribute access for different PyG versions
    if hasattr(gat_layer, 'lin_l'):
        enc_weights = gat_layer.lin_l.weight.detach().cpu().numpy()
    elif hasattr(gat_layer, 'lin_src'):
         enc_weights = gat_layer.lin_src.weight.detach().cpu().numpy()
    else:
        enc_weights = gat_layer.lin.weight.detach().cpu().numpy()

    sns.heatmap(enc_weights, ax=axes[0,0], cmap="coolwarm", center=0)
    axes[0,0].set_title(f"Node Features (Encoder)\nInput ({enc_weights.shape[1]}) $\\rightarrow$ Emb ({enc_weights.shape[0]})", fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel("Node Features", fontsize=12)
    axes[0,0].set_xticklabels(['LeftConn', 'RightConn'], fontsize=10)
    axes[0,0].set_ylabel("Embedding Neurons", fontsize=12)

    # 1.2 Decoder: Layer 1 (Hidden)
    dec_l1 = model.decoder[0].weight.detach().cpu().numpy()
    
    sns.heatmap(dec_l1, ax=axes[0,1], cmap="coolwarm", center=0)
    axes[0,1].set_title(f"Hidden Layer (Decoder)\nEmb ({dec_l1.shape[1]}) $\\rightarrow$ Hidden ({dec_l1.shape[0]})", fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel("Hidden Neurons", fontsize=12)
    axes[0,1].set_xlabel("Embedding Neurons", fontsize=12)

    # 1.3 Decoder: Output Layer (Q-Values)
    dec_out = model.decoder[2].weight.detach().cpu().numpy().T 
    
    sns.heatmap(dec_out, ax=axes[0,2], cmap="coolwarm", center=0, annot=False)
    axes[0,2].set_title("Output Layer (Decoder)\nHidden $\\rightarrow$ Q-Values", fontsize=14, fontweight='bold')
    axes[0,2].set_xlabel("Action Q-Values", fontsize=12)
    axes[0,2].set_ylabel("Hidden Neurons", fontsize=12)
    axes[0,2].set_xticklabels(['Q(Entangle)', 'Q(Swap)'], rotation=0, fontsize=10) 

    # --- ROW 2: EDGE FEATURES (Continuous [0, 1]) ---
    
    # Create the continuous edge feature sweep (0 to 1)
    edge_steps = 100
    edge_x = np.linspace(0, 1, edge_steps) # X-axis for contour plots
    
    # 2.1 Encoder Edge Projection
    # Check for edge projection weights in GATv2 (usually lin_edge)
    if hasattr(gat_layer, 'lin_edge'):
        # Shape: [Out_Channels * Heads, Edge_Dim]
        edge_weight_matrix = gat_layer.lin_edge.weight.detach().cpu().numpy()
    else:
        # Fallback if specific attribute missing, create dummy zeros to prevent crash
        print("Warning: No 'lin_edge' found in GAT layer. Plotting zeros.")
        edge_weight_matrix = np.zeros((enc_weights.shape[0], 1))

    # Calculate contribution: Weight * Feature_Value
    # Shape: [Emb_Dim, 1] @ [1, 100] -> [Emb_Dim, 100]
    # This represents the embedding activation solely due to the edge feature
    enc_edge_contrib = edge_weight_matrix @ edge_x.reshape(1, -1)
    

    # This ensures 0 is exactly "white"
    norm_enc = mcolors.TwoSlopeNorm(vmin=enc_edge_contrib.min(), vcenter=0, vmax=enc_edge_contrib.max())
    # Plot Contour
    # Y-axis: Embedding Neurons, X-axis: Edge Value
    cnt0 = axes[1,0].contourf(edge_x, np.arange(enc_edge_contrib.shape[0]), enc_edge_contrib, 
                              levels=20, center = 0, cmap="coolwarm", 
                              norm=norm_enc)
    axes[1,0].set_title(f"Edge Feature $\\rightarrow$ Embedding\nSensitivity Scan", fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel("Edge Feature Value", fontsize=12)
    axes[1,0].set_ylabel("Embedding Neurons", fontsize=12)
    plt.colorbar(cnt0, ax=axes[1,0])

    # 2.2 Propagate to Decoder Hidden Layer
    # We take the encoder output (enc_edge_contrib) and multiply by Decoder L1 weights
    # Shape: [Hidden, Emb] @ [Emb, 100] -> [Hidden, 100]
    hidden_edge_contrib = dec_l1 @ enc_edge_contrib

    norm_hid = mcolors.TwoSlopeNorm(vmin=hidden_edge_contrib.min(), vcenter=0, vmax=hidden_edge_contrib.max())
    
    cnt1 = axes[1,1].contourf(edge_x, np.arange(hidden_edge_contrib.shape[0]), hidden_edge_contrib, 
                              levels=20, cmap="coolwarm", norm=norm_hid)
    axes[1,1].set_title(f"Edge Effect $\\rightarrow$ Hidden Layer\n(Propagated)", fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel("Edge Feature Value", fontsize=12)
    axes[1,1].set_ylabel("Hidden Neurons", fontsize=12)
    plt.colorbar(cnt1, ax=axes[1,1])

    # 2.3 Propagate to Output (Rotated)
    # Calculate contribution: [Q, Hidden] @ [Hidden, 100] -> [Q, 100]
    q_edge_contrib = dec_out.T @ hidden_edge_contrib

    axes[1,2].plot(edge_x, q_edge_contrib[0, :], linestyle= 'dotted', label='Q(Entangle)', color='black', linewidth=2.5)
    
    # Plot Q(Swap) - Usually index 1
    axes[1,2].plot(edge_x, q_edge_contrib[1, :], linestyle='solid', label='Q(Swap)', color='black', linewidth=2.5)
    
    axes[1,2].set_title(f"Edge Feature Sensitivity)", fontsize=14, fontweight='bold')
    axes[1,2].set_xlabel("Edge Feature Value", fontsize=12)
    axes[1,2].set_ylabel("Contribution to Q-Value", fontsize=12)
    
    # Add a horizontal zero line for reference
    axes[1,2].axhline(0, color='black', linestyle='--', alpha=0.5)
    
    axes[1,2].legend(fontsize=12)
    axes[1,2].grid(True, alpha=0.3)

    # Final Layout Adjustments
    # Add Row Labels
    axes[0,0].text(-0.25, 0.5, 'Node Feature Weights\n', 
                   transform=axes[0,0].transAxes, 
                   va='center', ha='center', rotation='vertical', 
                   fontsize=16, fontweight='bold')

    axes[1,0].text(-0.25, 0.5, 'Edge Feature Flow\n', 
                   transform=axes[1,0].transAxes, 
                   va='center', ha='center', rotation='vertical', 
                   fontsize=16, fontweight='bold')

    if savefig:
        plt.savefig(save_img_dir + "weights_and_edge_contours.png", bbox_inches='tight')
    
    plt.show()
    plt.close()


    # 
    # 
    # 
    # 
    # --- 3. Visualize Model Output (Policy Check) ---
    print("\n--- Visualizing Policy Response on Diagnostic States ---")
    
    # Create Diagnostic States
    states_data = {}
    
    # State A: Empty (Should prefer Entangle)
    env = RepeaterNetwork(n=n_nodes)
    states_data['Empty'] = env.tensorState()
    
    # State B: Full Chain (Should prefer Swap on internal nodes)
    env = RepeaterNetwork(n=n_nodes)
    # Manually set all links to high fidelity
    for i in range(n_nodes - 1):
        env.setLink((i, i+1), 1.0, linkType=1)
    states_data['Full'] = env.tensorState()

    # State B: Full Chain (Should prefer NOT TO SWAP)
    env = RepeaterNetwork(n=n_nodes)
    # Manually set all links to high fidelity
    for i in range(n_nodes - 1):
        env.setLink((i, i+1), 0.0001, linkType=1)
    states_data['Full Weak'] = env.tensorState()
    
    # State C: Broken Chain (Gap at index 1)
    env = RepeaterNetwork(n=n_nodes)
    for i in range(n_nodes - 1):
        if i != 1: # Leave a gap
            env.setLink((i, i+1), 1.0, linkType=1)
    states_data['Full no 1-2'] = env.tensorState()

    #State D: Almost there
    env = RepeaterNetwork(n=n_nodes)
    env.setLink((0,1), 1)
    env.setLink((1,env.n-1), np.exp(-cutoff_tau_ratio))
    states_data['S1 wins'] = env.tensorState()

    #State D: Swap death
    env = RepeaterNetwork(n=n_nodes)
    env.setLink((0,1), np.exp(-cutoff_tau_ratio))
    env.setLink((1,env.n-1), 0.001)
    states_data['S1 loses (expires)'] = env.tensorState()


    # Run Inference
    records = []
    with torch.no_grad():
        for label, data in states_data.items():
            data = data.to(device)
            # Ensure edge_attr is 2D as per your fix
            if data.edge_attr is not None and data.edge_attr.dim() == 1:
                data.edge_attr = data.edge_attr.view(-1, 1)
                
            q_values = model(data).cpu().numpy() # [N, 2]
            
            for node in range(n_nodes-1):
                q_ent = q_values[node, 0]
                q_swap = q_values[node, 1]
                diff = q_swap - q_ent # Positive means Swap preferred
                
                records.append({
                    'State': label,
                    'Node': node,
                    'Q(Entangle)': q_ent,
                    'Q(Swap)': q_swap,
                    'Preference (Swap - Ent)': diff
                })

    df = pd.DataFrame(records)

    # Plot Policy Preferences
    # We plot Q(Swap) - Q(Entangle). 
    # Green (>0) = Swap, Red (<0) = Entangle
    
    plt.figure(figsize=(12, 6))
    pivot_df = df.pivot(index='State', columns='Node', values='Preference (Swap - Ent)')
    
    sns.heatmap(pivot_df, cmap="RdYlGn", center=0, annot=True, fmt=".2f", linewidths=.5)
    plt.title(f"Policy Preference Map (Green=Swap, Red=Entangle)\nN={n_nodes}", fontsize=18)
    plt.xlabel("Node Index", fontsize=16)
    plt.ylabel("Input State Scenario", fontsize=16)
    plt.savefig(save_img_dir + "policy_preference.png") if savefig else None
    plt.show()
    plt.close
    
    # Plot Raw Q-values for details
    g = sns.catplot(
        data=df.melt(id_vars=['State', 'Node'], value_vars=['Q(Entangle)', 'Q(Swap)'], var_name='Action', value_name='Q-Value'),
        x='Node', y='Q-Value', hue='Action', col='State',
        kind='bar', height=4, aspect=1, sharey=False, col_wrap=3
    )
    g.set(ylim=(df[['Q(Entangle)', 'Q(Swap)']].min().min() - 0.05, None))
    g.figure.subplots_adjust(top=0.85)
    g.figure.suptitle("Raw Q-Values per Node", fontsize = 18)
    plt.savefig(save_img_dir + "q_values_per_node.png") if savefig else None
    plt.show()
    plt.close

if __name__ == '__main__':
    #visualize_gnn_diagnostics(model_path=QRNAgent().policy_net.state_dict(), n_nodes=4) # run this to get initialization results
    path = 'assets/trained_models/Rd(12-2)l4u4e2450m80p50a99t50c15/Rd(12-2)l4u4e2450m80p50a99t50c15.pth'
    visualize_gnn_diagnostics(model_path=path, 
                              n_nodes=4, 
                              cutoff_tau_ratio=0.3,
                              savefig=True)