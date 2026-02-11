import torch
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.pyplot as plt
import pandas as pd
from base.model import GNN
from base.repeaters import RepeaterNetwork
from base.agent import QRNAgent

def visualize_gnn_diagnostics(model_path, n_nodes=4, cutoff_tau_ratio = 1, savefig=True):
    """
    Visualizes GNN weights and analyzes model behavior on diagnostic states.
    
    Args:
        model_path (str or dict): Path to .pth file or state_dict object.
        n_nodes (int): Number of nodes for the diagnostic states.
    """
    # --- 1. Load Model ---
    save_img_dir = model_path.rsplit('/', 1)[0] + '/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN(node_dim=2, output_dim=2).to(device)
    
    if isinstance(model_path, str):
        weights = torch.load(model_path, map_location=device)
    else:
        weights = model_path
        
    model.load_state_dict(weights)
    model.eval()

    # --- 2. Visualize Weights (Heatmaps) ---
    print("--- Visualizing Learned Weights ---")
    fig_w, axes_w = plt.subplots(1, 3, figsize=(18, 5))
    
    # Encoder: GATv2 Projection (Source Nodes)
    # Projects input features [Node_dim] -> [Embedding_dim * Heads]
    # Note: Accessing internal parameters depends on PyG version, usually 'lin_l' or 'lin_src'
    gat_layer = model.encoder[0]
    if hasattr(gat_layer, 'lin_l'):
        enc_weights = gat_layer.lin_l.weight.detach().cpu().numpy()
    elif hasattr(gat_layer, 'lin_src'):
         enc_weights = gat_layer.lin_src.weight.detach().cpu().numpy()
    else:
        # Fallback for some versions: lin is the linear layer
        enc_weights = gat_layer.lin.weight.detach().cpu().numpy()

    sns.heatmap(enc_weights, ax=axes_w[0], cmap="coolwarm", center=0)
    axes_w[0].set_title(f"GATv2 Encoder Weights\nInput ({enc_weights.shape[1]}) -> Emb ({enc_weights.shape[0]})")
    axes_w[0].set_xlabel("Input Features")
    axes_w[0].set_xticklabels(['LeftConnection', 'RightConnection'])
    axes_w[0].set_ylabel("Embedding Dimensions")

    # Decoder: Layer 1 (Hidden)
    dec_l1 = model.decoder[0].weight.detach().cpu().numpy()
    sns.heatmap(dec_l1, ax=axes_w[1], cmap="coolwarm", center=0)
    axes_w[1].set_title(f"Decoder Hidden Layer\nEmb ({dec_l1.shape[1]}) -> Hidden ({dec_l1.shape[0]})")

    # Decoder: Output Layer (Q-Values)
    # --- Plot D: Decoder Output Layer (Transposed) ---
    # Transpose (.T) so Q-values are on the X-axis
    dec_out = model.decoder[2].weight.detach().cpu().numpy().T 
    
    sns.heatmap(dec_out, ax=axes_w[2], cmap="coolwarm", center=0, annot=False)
    axes_w[2].set_title("Decoder Output Layer\nHidden -> Q-Values")
    axes_w[2].set_xlabel("Action Q-Values")
    axes_w[2].set_ylabel("Hidden Neurons")
    # Rotation 0 makes them horizontal and readable
    axes_w[2].set_xticklabels(['Q(Entangle)', 'Q(Swap)'], rotation=0) 
    
    plt.tight_layout()
    plt.savefig(save_img_dir + "weights.png") if savefig else None
    plt.show()
    plt.close

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
    
    # State C: Broken Chain (Gap at index 1)
    env = RepeaterNetwork(n=n_nodes)
    for i in range(n_nodes - 1):
        if i != 1: # Leave a gap
            env.setLink((i, i+1), 1.0, linkType=1)
    states_data['Full no 1->2'] = env.tensorState()

    #State D: Almost there
    env = RepeaterNetwork(n=n_nodes)
    env.setLink((0,1), 1)
    env.setLink((1,env.n-1), 1)
    states_data['S1 wins'] = env.tensorState()


    # Run Inference
    records = []
    with torch.no_grad():
        for label, data in states_data.items():
            data = data.to(device)
            # Ensure edge_attr is 2D as per your fix
            if data.edge_attr is not None and data.edge_attr.dim() == 1:
                data.edge_attr = data.edge_attr.view(-1, 1)
                
            q_values = model(data).cpu().numpy() # [N, 2]
            
            for node in range(n_nodes):
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
    plt.title(f"Policy Preference Map (Green=Swap, Red=Entangle)\nN={n_nodes}")
    plt.xlabel("Node Index")
    plt.ylabel("Input State Scenario")
    plt.savefig(save_img_dir + "policy_preference.png") if savefig else None
    plt.show()
    plt.close
    
    # Plot Raw Q-values for details
    g = sns.catplot(
        data=df.melt(id_vars=['State', 'Node'], value_vars=['Q(Entangle)', 'Q(Swap)'], var_name='Action', value_name='Q-Value'),
        x='Node', y='Q-Value', hue='Action', col='State',
        kind='bar', height=4, aspect=1, sharey=False
    )
    g.set(ylim=(df[['Q(Entangle)', 'Q(Swap)']].min().min() - 0.05, None))
    g.figure.subplots_adjust(top=0.85)
    g.figure.suptitle("Raw Q-Values per Node")
    plt.savefig(save_img_dir + "q_values_per_node.png") if savefig else None
    plt.show()
    plt.close

if __name__ == '__main__':
    #visualize_gnn_diagnostics(model_path=QRNAgent().policy_net.state_dict(), n_nodes=4) # run this to get initialization results
    visualize_gnn_diagnostics(model_path="./assets/trained_models/d(11-2)l4u6e1000m300p.8a95t100c80/d(11-2)l4u6e1000m300p.8a95t100c80.pth", n_nodes=4, savefig=True)