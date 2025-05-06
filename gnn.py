#!/usr/bin/env python
"""
hybrid_gnn_llm_pipeline.py

An all-in-one Python script implementing a hybrid GNN + LLM embedding pipeline for materials property prediction.
Uses PyTorch Geometric for graph neural network modeling and Hugging Face Transformers for LLM-based text embeddings.

Dependencies:
    - torch
    - torch-geometric and related packages (e.g., torch-scatter, torch-sparse, etc.)
    - transformers

Usage:
    python hybrid_gnn_llm_pipeline.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# PyTorch Geometric imports
from torch_geometric.data import Data, DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# Hugging Face Transformers for LLM text embeddings
from transformers import AutoTokenizer, AutoModel

# ------------------------------
# 1. Dummy Dataset Creation
# ------------------------------
class DummyCrystalGraphDataset(torch.utils.data.Dataset):
    """
    A dummy dataset that returns:
      - A PyTorch Geometric Data object containing a fully connected graph with random node features.
      - A dummy text description for the crystal.
      - A random float label as the target property.
    """
    def __init__(self, num_samples=50, num_nodes=5):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.data_list = []
        self.texts = []
        self.labels = []
        for i in range(num_samples):
            # Generate random node features (e.g., atomic coordinates, here 3 dimensions per node)
            x = torch.randn(num_nodes, 3)
            # Create a fully connected graph (excluding self-loops)
            edge_index = self.create_full_edge_index(num_nodes)
            data = Data(x=x, edge_index=edge_index)
            self.data_list.append(data)
            # Dummy textual description (in practice use Robocystallographer/ChemNLP for domain-specific description)
            self.texts.append(f"This is sample {i} describing a crystal with random properties.")
            # Random label as the target property
            self.labels.append(random.uniform(0, 5))

    def create_full_edge_index(self, num_nodes):
        edge_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_list.append([i, j])
        # edge_index shape should be [2, num_edges]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_list[idx], self.texts[idx], torch.tensor(self.labels[idx], dtype=torch.float)

# ------------------------------
# 2. GNN Module using PyTorch Geometric
# ------------------------------
class GNNModule(nn.Module):
    """
    A simple GNN using two GCNConv layers.
    It takes a graph (Data object with node features and edge_index) and returns a graph-level embedding
    via global mean pooling.
    """
    def __init__(self, in_channels=3, hidden_channels=64, out_channels=128):
        super(GNNModule, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, data):
        """
        Expects a batch of graph data with attributes: x, edge_index, and batch.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        # Global mean pooling to obtain a fixed-size graph-level embedding
        x = global_mean_pool(x, batch)
        return x  # Shape: (batch_size, out_channels)

# ------------------------------
# 3. LLM Embedding Module using Transformers
# ------------------------------
class LLMEmbedder:
    """
    Uses a pretrained transformer (e.g., 'bert-base-uncased') to extract text embeddings.
    Embeddings are obtained by mean pooling the token embeddings of the last hidden state.
    """
    def __init__(self, model_name="bert-base-uncased", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def get_text_embedding(self, text_list):
        encoded = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoded)
            last_hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
            attention_mask = encoded["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)
            # Mean pooling over the valid tokens
            sum_embeddings = torch.sum(last_hidden_states * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            pooled_embedding = sum_embeddings / sum_mask
        return pooled_embedding  # Shape: (batch, hidden_dim)

# ------------------------------
# 4. Hybrid Model: Concatenating GNN + LLM Embeddings and MLP Regressor
# ------------------------------
class HybridModel(nn.Module):
    """
    The hybrid model first extracts graph-level embeddings using the GNNModule
    and text embeddings using the LLMEmbedder. These embeddings are concatenated
    and passed through an MLP to predict the target property.
    """
    def __init__(self, gnn_module, llm_embed_dim=768, gnn_embed_dim=128, hidden_dim=256):
        super(HybridModel, self).__init__()
        self.gnn = gnn_module
        self.concat_dim = llm_embed_dim + gnn_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data, text_emb):
        # Extract graph embedding from GNN
        gnn_embed = self.gnn(data)  # Shape: (batch_size, gnn_embed_dim)
        # Concatenate with LLM text embedding
        combined = torch.cat([gnn_embed, text_emb], dim=1)
        output = self.mlp(combined)  # Regression output
        return output.squeeze(1)  # Shape: (batch_size,)

# ------------------------------
# 5. Training and Evaluation Pipeline
# ------------------------------
def train(model, dataloader, llm_embedder, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for data, texts, labels in dataloader:
        # Move graph data and labels to device
        data = data.to(device)
        labels = labels.to(device)
        # Get LLM embeddings for the batch of texts
        text_emb = llm_embedder.get_text_embedding(texts)
        optimizer.zero_grad()
        # Forward pass through the hybrid model
        predictions = model(data, text_emb)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, llm_embedder, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, texts, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)
            text_emb = llm_embedder.get_text_embedding(texts)
            predictions = model(data, text_emb)
            loss = criterion(predictions, labels)
            total_loss += loss.item() * labels.size(0)
            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())
    avg_loss = total_loss / len(dataloader.dataset)
    # Compute Mean Absolute Error (MAE) as an additional metric
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    mae = torch.mean(torch.abs(all_preds - all_labels))
    return avg_loss, mae.item()

# ------------------------------
# 6. Main Function
# ------------------------------
def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the dummy dataset
    dataset = DummyCrystalGraphDataset(num_samples=100, num_nodes=5)

    # Split dataset into train and test (80/20 split)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Use PyTorch Geometric's DataLoader
    train_loader = GeoDataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = GeoDataLoader(test_dataset, batch_size=8, shuffle=False)

    # Initialize the GNN module (using PyTorch Geometric)
    gnn_module = GNNModule(in_channels=3, hidden_channels=64, out_channels=128)
    
    # Initialize the LLM embedder (using a pretrained BERT model; you may switch to a domain-specific model)
    llm_embedder = LLMEmbedder(model_name="bert-base-uncased", device=device)

    # Build the hybrid model, concatenating GNN and LLM embeddings
    model = HybridModel(gnn_module, llm_embed_dim=768, gnn_embed_dim=128, hidden_dim=256).to(device)

    # Set up optimizer and loss function (MSE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, llm_embedder, optimizer, criterion, device)
        test_loss, test_mae = evaluate(model, test_loader, llm_embedder, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} | Test Loss (MSE): {test_loss:.4f} | Test MAE: {test_mae:.4f}")

    print("Training and evaluation complete.")

if __name__ == "__main__":
    main()
