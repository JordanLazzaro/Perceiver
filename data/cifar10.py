import torch
import torchvision
import torchvision.transforms as transforms


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda x: x.view(-1, 3))
        ])

    def prepare_data(self):
        torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    def setup(self, stage=None):
        if stage in ('fit', None):
            self.cifar10_train = torchvision.datasets.CIFAR10(
                root='./data', train=True, transform=self.transform)
            self.cifar10_val = torchvision.datasets.CIFAR10(
                root='./data', train=False, transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar10_val, batch_size=self.batch_size, shuffle=False, num_workers=2)