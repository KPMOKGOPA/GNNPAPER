import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from torch_geometric.nn import global_mean_pool
from sklearn.manifold import TSNE


def evaluate_model(model, loader, device='cpu'):
    model.eval()
    all_preds, all_labels, all_probs, all_u = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            u_list = [g.u for g in batch.to_data_list() if hasattr(g, 'u') and g.u is not None]
            u = torch.stack(u_list).float() if len(u_list) > 0 else None

            x = F.relu(model.conv1(batch.x, batch.edge_index))
            x = F.relu(model.conv2(x, batch.edge_index))
            x = global_mean_pool(x, batch.batch)

            if u is not None:
                x = torch.cat([x, u], dim=-1)

            pred = torch.sigmoid(model.lin2(F.relu(model.lin1(x)))).squeeze()
            all_probs.append(pred.cpu())
            all_preds.append((pred > 0.5).float().cpu())
            all_labels.append(batch.y.cpu())
            if u is not None:
                all_u.append(u.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    all_u = torch.cat(all_u) if all_u else None

    return all_preds, all_labels, all_probs, all_u


if __name__ == '__main__':
    model = ReactionGNN(
        in_channels=11,
        hidden_channels=64,
        global_dim=0
        )    

    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=30,
        lr=1e-3,
        device=device
        )

    train_metrics = evaluate_model(
        model,
        train_loader,
        device=device
        )

    preds, labels, probs, u_features = evaluate_model(model, val_loader)
    pass

