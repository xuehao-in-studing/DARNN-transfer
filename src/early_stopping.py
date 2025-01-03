#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2023/5/13 10:42
# @Author : 朱
# @FileName : early_stopping.py
import os
import numpy as np
import torch
from arguments import parse_args


args = parse_args(None)
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, save: bool = True, path=f'../models/GZ_{args.object_col}.pt', trace_func=print):
        """

        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
            path (str): Path for the checkpoint to be saved to. Default: '../models/checkpoint.pt'
            trace_func (function): trace print function. Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.save = save
        self.trace_func = trace_func
        if not os.path.exists("../models"):
            os.mkdir("../models")

    def __call__(self, val_loss, model):
        """
        Checks if the validation loss has improved, and if not, increments the early stop counter.
        If the early stop counter exceeds patience, training is stopped.

        Args:
            val_loss (float): The current validation loss.
            model (tf.keras.Model): The model being trained.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'The best score is {self.val_loss_min:.6f}, now is {val_loss:.6f},\n'
                            f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves models when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving models ...')
        if self.save:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


if __name__ == '__main__':
    # model = torchvision.models.alexnet(pretrained=True)
    # val_loss = [0.6, 0.5, 0.4, 0.41, 0.3, 0.32, 0.2, 0.21, 0.22, 0.196, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28]
    # early_stopping = EarlyStopping(patience=5, verbose=True, path="../../models/checkpoint.pt")
    #
    # for loss in val_loss:
    #     early_stopping(loss, model)
    #     if early_stopping.early_stop:
    #         print("Early stopping")
    #         break
    pass
