
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_prop.graphs_generation import ReactionGraphDataset
from torch_geometric.nn import GraphConv, global_mean_pool


def build_dataloaders(df_main, batch_size=16, val_split=0.2, shuffle_train=True, seed=42):
    dataset = ReactionGraphDataset(df_main)
    
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val],
                                                               generator=torch.Generator().manual_seed(seed))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

class ReactionGNN(nn.Module):
    def __init__(self, in_channels=11, hidden_channels=64, out_channels=1, global_dim=2):
        super().__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels + global_dim, 64)
        self.lin2 = nn.Linear(64, out_channels)

    def forward(self, x, edge_index, batch, u=None):
        # Node-level graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Pool to graph-level embedding
        x = global_mean_pool(x, batch)  

        # Concatenate global 
        if u is not None:
            x = torch.cat([x, u], dim=-1)  

        # MLP readout
        x = F.relu(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))  # Binary classification
        return x.view(-1) 


def train_model(model, train_loader, val_loader,
                epochs=30, lr=1e-3, device='cpu'):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    model.to(device)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        #training
        model.train()

        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:

            batch = batch.to(device)

            u_list = [g.u for g in batch.to_data_list()
                      if g.u is not None]

            u = torch.stack(u_list).to(device) if len(u_list) > 0 else None

            optimizer.zero_grad()

            pred = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                u=u
            )

            loss = criterion(pred, batch.y.float())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pred_class = (pred > 0.5).float()

            correct += (pred_class == batch.y).sum().item()
            total += batch.y.numel()

        train_loss /= len(train_loader)
        train_acc = correct / total
        #validation
        model.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for batch in val_loader:

                batch = batch.to(device)

                u_list = [g.u for g in batch.to_data_list()
                          if g.u is not None]

                u = torch.stack(u_list).to(device) if len(u_list) > 0 else None

                pred = model(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    u=u
                )

                loss = criterion(pred, batch.y.float())

                val_loss += loss.item()

                pred_class = (pred > 0.5).float()

                correct += (pred_class == batch.y).sum().item()
                total += batch.y.numel()

        val_loss /= len(val_loader)
        val_acc = correct / total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"Train Loss={train_loss:.4f} | "
            f"Train Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Val Acc={val_acc:.4f}"
        )

    return history

def evaluate_model(model, loader, device='cpu'):
    model.eval()

    criterion = nn.BCELoss()

    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:

            batch = batch.to(device)

            u_list = [g.u for g in batch.to_data_list() if g.u is not None]
            u = torch.stack(u_list).to(device) if len(u_list) > 0 else None

            probs = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                u=u
            )

            labels = batch.y.float().view(-1)
            probs = probs.view(-1)

            loss = criterion(probs, labels)
            total_loss += loss.item()

            preds = (probs > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.numel()

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(loader)
    acc = correct / total

    all_probs = torch.cat(all_probs)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    print(
        f"Loss: {avg_loss:.4f} | "
        f"Accuracy: {acc:.4f}"
    )

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "preds": all_preds,
        "probs": all_probs,
        "labels": all_labels
    }
