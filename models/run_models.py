
import torch
from models.models import build_dataloaders, ReactionGNN, train_model, evaluate_model


def main(df_main):
    train_loader, val_loader = build_dataloaders(df_main, batch_size=16, val_split=0.2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ReactionGNN(in_channels=11, hidden_channels=64, global_dim=0)

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

    val_metrics = evaluate_model(
        model,
        val_loader,
        device=device
    )

    print("Train metrics:", train_metrics)
    print("Validation metrics:", val_metrics)
    return history, train_metrics, val_metrics


if __name__ == '__main__':
    df_main = None
    if df_main is None:
        raise RuntimeError("Please define `df_main` as a pandas DataFrame before running run_models.py")
    main(df_main)
