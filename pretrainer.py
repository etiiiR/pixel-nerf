from solo.methods import SimCLR
from solo.data.classification_dataloader import prepare_data
from solo.args.setup import parse_args_pretrain

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
import os

# Set up args manually
class Args:
    # Model
    method = "simclr"
    backbone = "resnet34"
    pretrained_feature_extractor = False

    # Data
    data_dir = "data/pollen_simclr"
    dataset = "custom"
    unique_classes = True

    # Training
    max_epochs = 100
    batch_size = 128
    num_workers = 4
    optimizer = "adam"
    precision = 16
    lr = 3e-4

    # Logging
    name = "simclr-pollen"
    log_every_n_steps = 10
    gpus = 1

    # SimCLR specific
    temperature = 0.2
    proj_hidden_dim = 2048
    proj_output_dim = 256

args = Args()

# Prepare data
train_loader = prepare_data(args)

# Create model
model = SimCLR(args)

# Train
trainer = Trainer(
    max_epochs=args.max_epochs,
    logger=CSVLogger("logs/", name=args.name),
    devices=1,
    accelerator="gpu",
)

trainer.fit(model, train_loader)