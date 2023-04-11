import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner import Tuner

from data.cifar10 import CIFAR10DataModule
from model import PerceiverClassificationHead
from lit_model import CIFAR10Classifier


input_seq_len = 1024 # M in paper, 32*32 for CIFAR-10
in_channels = 3 # kv_dim in huggingface impl.
pos_emb_channels = in_channels * 2

latent_seq_len = 128 # N in paper
latent_channels = 256 # q_dim in huggingface impl.

out_channels = latent_channels

nheads = 8
nxheads = 1
nlayers = 4
nblocks = 2

batch_size = 128

################
# Prepare data #
################
cifar10_data = CIFAR10DataModule(batch_size=batch_size)

#################
# Prepare model #
#################
model = PerceiverClassificationHead(
    latent_channels,
    latent_seq_len,
    in_channels,
    input_seq_len,
    out_channels,
    nheads,
    nxheads,
    nlayers,
    nblocks,
    pos_emb_channels,
    dropout=0.0
)

cifar10_classifier = CIFAR10Classifier(model)

###################
# Prepare Trainer #
###################
wandb_logger = WandbLogger(project="Perceiver CIFAR-10", log_model=True)
wandb_logger.watch(cifar10_classifier, log="all")

trainer = pl.Trainer(
    max_epochs=25,
    devices=1,
    accelerator="gpu",
    precision="16-mixed",
    logger=wandb_logger,
    # overfit_batches=1
)

tuner = Tuner(trainer)
tuner.lr_find(cifar10_classifier, datamodule=cifar10_data)

#############
# Pump Iron #
#############
trainer.fit(cifar10_classifier, cifar10_data)

wandb.finish()