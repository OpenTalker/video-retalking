import torch
import os
from PIL import Image
import random
import numpy as np
import pickle
import torchvision.transforms as transforms

from .celeba import CelebADataset


def create_dataloader(opt):
    data_loader = DataLoader()
    data_loader.initialize(opt)
    return data_loader


class DataLoader:
    def name(self):
        return self.dataset.name() + "_Loader"

    def create_datase(self):
        # specify which dataset to load here
        loaded_dataset = os.path.basename(self.opt.data_root.strip('/')).lower()
        if 'celeba' in loaded_dataset or 'emotion' in loaded_dataset:
            dataset = CelebADataset()
        else:
            dataset = BaseDataset()
        dataset.initialize(self.opt)
        return dataset

    def initialize(self, opt):
        self.opt = opt
        self.dataset = self.create_datase()
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.n_threads)
        )

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
