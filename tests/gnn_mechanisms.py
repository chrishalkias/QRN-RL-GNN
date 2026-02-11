import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from base.model import GNN
from base.repeaters import RepeaterNetwork

def visualize_gnn_mechanisms(model_path, 
                             n_nodes=4, 
                             cutoff_tau_ratio = 1,
                             pca_samples = 500, 
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
    # PLOT 1: Decision Phase Diagram (Sensitivity Analysis)
    # =========================================================================
    # Hypothesis: The agent should only SWAP if both Left and Right fidelities 
    # are above a certain threshold. We will map this threshold.
    
    print("Generating Decision Phase Diagram...")
    resolution = 20
    fidelities = np.linspace(0, 1.0, resolution)
    
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
            env.setLink((1, 2), f_right, linkType=1)
            
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
    ax = sns.heatmap(sensitivity_grid, xticklabels=np.round(fidelities, 1), 
                     yticklabels=np.round(fidelities, 1), cmap="RdYlGn", center=0)
    
    ax.invert_yaxis() # Ensure 0 is at bottom
    plt.title(f"Decision Phase Diagram (Node {target_node})\nDoes the agent want to Swap?", fontsize=14)
    plt.xlabel(f"Fidelity of Right Link ({target_node}-{target_node+1})", fontsize=12)
    plt.ylabel(f"Fidelity of Left Link ({target_node-1}-{target_node})", fontsize=12)

    # --- ADDED: Theoretical Boundary Line (F1 * F2 = e^-0.5) ---
    threshold_val = np.exp(-cutoff_tau_ratio)
    
    # 1. Generate X coordinates in "Heatmap Index Space" (0 to resolution-1)
    # We start slightly above 0 to avoid division by zero errors
    x_indices = np.linspace(0.1, resolution , 200)
    
    # 2. Convert indices to physical values (0.0 to 1.0)
    x_phys = x_indices / (resolution - 1)
    
    # 3. Calculate physical Y using the equation: y = threshold / x
    y_phys = threshold_val / x_phys
    
    # 4. Convert physical Y back to "Heatmap Index Space"
    y_indices = y_phys * (resolution - 1)
    
    # 5. Plot the line
    # We restrict ylim to keep the plot neat (ignoring the asymptote)
    plt.plot(x_indices, y_indices, color='cyan', linestyle='--', linewidth=2.5, label=r'$F_1 \cdot F_2 = e^{-t_{cutoff}/\tau}$')
    plt.legend(loc='upper right', frameon=True, facecolor='black', labelcolor='white')
    plt.ylim(0, resolution - 1)
    
    # Add a contour line for the Decision Boundary (where Preference = 0)
    # We need to map grid indices to plot coordinates
    cnt = plt.contour(np.arange(0.5, resolution), np.arange(0.5, resolution), 
                      sensitivity_grid, levels=[0], colors='black', linewidths=2, linestyles='--')
    plt.clabel(cnt, inline=True, fontsize=10, fmt='Decision Boundary')
    
    plt.tight_layout()
    plt.savefig(save_dir +"phase_diagram.png") if savefig else None
    plt.show()
    plt.close()

    # =========================================================================
    # PLOT 2: Latent Space Visualization (PCA)
    # =========================================================================
    # We generate random states, extract the GNN's internal embeddings,
    # and see if the model mathematically separates "Swap" states from "Entangle" states.
    
    print("Generating Latent Space PCA...")
    n_samples = pca_samples
    embeddings_list = []
    labels_list = []
    
    # Hook to capture output of the Encoder (GATv2)
    # The model structure is: Encoder -> Decoder -> Output
    # We want the output of model.encoder
    def get_embeddings(data):
        with torch.no_grad():
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            # Manually run encoder layers
            for layer in model.encoder:
                # Handle GATv2Conv signature
                x = layer(x, edge_index, edge_attr=edge_attr)
            return x # This is the embedding [N_nodes, Embedding_dim]

    for _ in range(n_samples):
        # Generate random state (p_entangle=1.0 ensures connections exist but have variable fidelity if we tweak)
        # Here we manually randomize fidelities to get a diverse dataset
        env = RepeaterNetwork(n=n_nodes)
        
        # Randomize links
        for i in range(n_nodes-1):
            if np.random.rand() > 0.3: # 70% chance of link existence
                fid = np.random.rand() # Random fidelity 0-1
                env.setLink((i, i+1), fid, linkType=1)
        
        data = env.tensorState().to(device)
        if data.edge_attr is not None and data.edge_attr.dim() == 1:
            data.edge_attr = data.edge_attr.view(-1, 1)
            
        # Get Embeddings & Decisions
        emb = get_embeddings(data).cpu().numpy() # [N, 32] (or whatever hidden dim is)
        q_vals = model(data).cpu().detach().numpy()       # [N, 2]
        
        # For each node, store its embedding and its preferred action
        for node in range(n_nodes):
            # Filter: only care about internal nodes (endpoints usually have fixed policies)
            if 0 < node < n_nodes - 1:
                embeddings_list.append(emb[node])
                
                # Determine Label
                # 0=Entangle, 1=Swap
                # We can also use "Strong Entangle" vs "Weak Swap" based on Q-diff
                action = np.argmax(q_vals[node])
                labels_list.append("Swap" if action == 1 else "Entangle")

    # Run PCA
    if len(embeddings_list) > 0:
        X = np.array(embeddings_list)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        df_pca = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Decision': labels_list
        })

        #add a small amount of noise for visualization (seperate overlapping points)
        jitter_strength = 0.05
        df_pca['PC1'] += np.random.normal(0, jitter_strength, size=len(df_pca))
        df_pca['PC2'] += np.random.normal(0, jitter_strength, size=len(df_pca))

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Decision', 
                        style='Decision', palette={'Swap': 'green', 'Entangle': 'red'}, s=60, alpha=0.7)
        
        plt.title(f"GNN Internal Representation (PCA)\nInternal Nodes (1 to {n_nodes-2})", fontsize=14)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title="Preferred Action")
        plt.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir+ "latent_pca.png") if savefig else None
        plt.show()
        plt.close()
    else:
        print("Not enough data for PCA.")

if __name__ == "__main__":
    # Example Usage
    visualize_gnn_mechanisms("./assets/trained_models/d(11-2)l4u4e4000m80p.5a99t50c15/d(11-2)l4u4e4000m80p.5a99t50c15.pth", 
                             n_nodes=8, 
                             cutoff_tau_ratio=0.5,
                             pca_samples = 1000, 
                             savefig=True)