
import pandas as pd
import torch

configs = [
    {"hidden": 32,  "lr": 1e-2},
    {"hidden": 32,  "lr": 5e-3},
    {"hidden": 32,  "lr": 1e-3},
    {"hidden": 32,  "lr": 5e-4},

    {"hidden": 64,  "lr": 1e-2},
    {"hidden": 64,  "lr": 5e-3},
    {"hidden": 64,  "lr": 1e-3},
    {"hidden": 64,  "lr": 5e-4},

    {"hidden": 128, "lr": 1e-2},
    {"hidden": 128, "lr": 5e-3},
    {"hidden": 128, "lr": 1e-3},
    {"hidden": 128, "lr": 5e-4},

    {"hidden": 256, "lr": 1e-2},
    {"hidden": 256, "lr": 5e-3},
    {"hidden": 256, "lr": 1e-3},
    {"hidden": 256, "lr": 5e-4},
]



train_loader, val_loader = build_dataloaders(
    df_main,
    batch_size=16,
    val_split=0.2
)

results = []

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for i, cfg in enumerate(configs):

    print("\n" + "="*60)
    print(f"Model {i+1}: {cfg}")
    print("="*60)

    model = ReactionGNN(
        in_channels=11,
        hidden_channels=cfg["hidden"],
        global_dim=0
    )

    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=15,
        lr=cfg["lr"],
        device=device
    )

    train_metrics = evaluate_model(
        model,
        train_loader,
        device=device
    )

    val_metrics = evaluate_model(
        model,
        val_loader,
        device=device
    )

    results.append({
        "hidden_channels": cfg["hidden"],
        "learning_rate": cfg["lr"],
        "train_acc": train_metrics["accuracy"],
        "val_acc": val_metrics["accuracy"],
        "train_loss": train_metrics["loss"],
        "val_loss": val_metrics["loss"]
    })

comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values(
    "val_acc",
    ascending=False
)

comparison_df

def analyze_robustness(model, loader, noise_levels=[0.01, 0.05, 0.1], device='cpu'):
    # Baseline accuracy
    baseline_preds, baseline_labels, _, _ = evaluate_model(model, loader, device)
    baseline_acc = (baseline_preds == baseline_labels).sum().item() / len(baseline_labels)
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    
    for noise in noise_levels:
        model.eval()
        noisy_preds, noisy_labels = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                # add Gaussian noise to node features
                noisy_x = batch.x + torch.randn_like(batch.x) * noise

                u_list = [g.u for g in batch.to_data_list() if hasattr(g,'u') and g.u is not None]
                u = torch.stack(u_list).float().to(device) if len(u_list) > 0 else None
                
                # Forward pass
                if u is not None:
                    pred = model(noisy_x, batch.edge_index, batch.batch, u=u)
                else:
                    pred = model(noisy_x, batch.edge_index, batch.batch)
                
                preds_batch = (torch.sigmoid(pred) > 0.5).float()
                noisy_preds.append(preds_batch.cpu())
                noisy_labels.append(batch.y.cpu())
        
        noisy_preds = torch.cat(noisy_preds)
        noisy_labels = torch.cat(noisy_labels)
        noisy_acc = (noisy_preds == noisy_labels).sum().item() / len(noisy_labels)
        print(f"Noise level {noise:.2f}: Accuracy = {noisy_acc:.4f} (Δ = {baseline_acc - noisy_acc:.4f})")

analyze_robustness(model, val_loader)

