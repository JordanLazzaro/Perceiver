import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from data.cifar10 import CIFAR10DataModule
from model import PerceiverClassificationHead
from lit_model import CIFAR10Classifier


input_seq_len = 1024 # M in paper, 32*32 for CIFAR-10
input_channels = 3 # kv_dim in huggingface impl.

latent_seq_len = 128 # N in paper
latent_channels = 512 # q_dim in huggingface impl.

out_channels = latent_channels

nheads = 4
nlayers = 12

batch_size = 64

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
    input_channels,
    input_seq_len,
    out_channels,
    nheads,
    nlayers
)

cifar10_classifier = CIFAR10Classifier(model)

###################
# Prepare Trainer #
###################
wandb_logger = WandbLogger(project="Perceiver CIFAR-10", log_model=True)
wandb_logger.watch(cifar10_classifier)

trainer = pl.Trainer(
    max_epochs=10,
    devices=1,
    accelerator="gpu",
    # precision="16-mixed",
    # accumulate_grad_batches=4,
    logger=wandb_logger
)

#############
# Pump Iron #
#############
trainer.fit(cifar10_classifier, cifar10_data)

wandb.finish()