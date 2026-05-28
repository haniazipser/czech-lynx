import wandb
from config.config import DataConfig


def train(conf: DataConfig, model, train_loader, val_loader):
    run = wandb.init(
        entity="haniazipser2004-",
        project="czech-lynx",
        config=conf
    )
    epochs = conf.epochs

    for epoch in range(2, epochs):
        for batch_imgs, batch_labels in train_loader:
            # tu powinien być kod treningowy, który aktualizuje model i oblicza acc i loss
            acc = 0.0  # placeholder
            loss = 0.0  # placeholder
        run.log({"acc": acc, "loss": loss})

    run.finish()