import os
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx

import torch
from torch.utils.data import DataLoader
from utils.train_utils import snapshot_model
from dataset import *


class AGEValidator:
    def __init__(self, loader, model, args, writer=None):
        self.args = args
        self.exp_args = args.experiment
        self.val_args = self.args.experiment.validation

        self.val_loader = loader
        self.model = model
        self.best_epoch = 0
        self.best_val_loss = math.inf
        self.val_losses = [math.inf]
        self.es_patience = self.val_args.es_patience * self.val_args.val_epochs
        self.writer = writer

    def validate(self, epoch):
        loss = self.compute_val_loss()
        if self.writer is not None:
            self.writer.add_scalar("Loss/Validation", loss, epoch)
        if loss < min(self.val_losses):
            self.args.experiment.best_val_epoch = epoch
            snapshot_model(self.model, epoch, self.args, is_best=True)
            self.best_epoch = epoch
            self.best_val_loss = loss
            self.val_losses.append(loss)
            return False
        self.val_losses.append(loss)
        if epoch - self.best_epoch >= self.es_patience + self.val_args.es_buffer:
            return True
        else:
            return False

    def compute_val_loss(self):
        self.model.eval()
        with torch.no_grad():
            loss = 0
            for batch in self.val_loader:
                loss += self.model(
                    batch
                ).mean()  # Loss already averaged within batch in fwd, we average over gpus here
        return loss.item() / len(self.val_loader)
