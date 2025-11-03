import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import random

# ----------------- Define GNN -----------------
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize model
model = GCN(in_channels=3, hidden_channels=4, out_channels=2)
model.eval()  # inference mode

# ----------------- Streamlit UI -----------------
st.title("Real-Time GNN Prediction Simulation")

num_nodes = st.slider("Number of nodes", 3, 10, 5)
simulate = st.button("Simulate Stream")

# ----------------- Real-Time Simulation -----------------
if simulate:
    predictions_text = st.empty()
    graph_fig = st.empty()

    for step in range(10):  # simulate 10 steps
        # Random node features
        nodes = [[random.random() for _ in range(3)] for _ in range(num_nodes)]
        
        # Simple edges: chain graph
        edges = [[i for i in range(num_nodes-1)] + [i for i in range(num_nodes-1)],
                 [i+1 for i in range(num_nodes-1)] + [i for i in range(1, num_nodes)]]

        x = torch.tensor(nodes, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)

        # GNN prediction
        with torch.no_grad():
            out = model(graph.x, graph.edge_index)
            preds = out.argmax(dim=1).tolist()

        # Update predictions in Streamlit
        predictions_text.text(f"Step {step+1} Predictions: {preds}")

        # Visualize graph
        G = nx.Graph()
        for i in range(num_nodes):
            G.add_node(i)
        for u, v in zip(edges[0], edges[1]):
            G.add_edge(u, v)

        plt.figure(figsize=(4,4))
        nx.draw(G, with_labels=True, node_color=preds, cmap=plt.cm.Set1, node_size=600)
        graph_fig.pyplot(plt)
        plt.close()

        st.sleep(1)  # simulate real-time stream
