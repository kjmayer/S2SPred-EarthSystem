"""Trainer modules for pytorch models.

Edited by: Kirsten Mayer
Written by: Elizabeth A. Barnes

Classes
---------
Trainer()

"""
import numpy as np
import torch
from utils.utils import MetricTracker
from base.base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metric_funcs,
        optimizer,
        scheduler,
        max_epochs,
        data,
        validation_data,
        device,
        config,
    ):
        super().__init__(
            model,
            criterion,
            metric_funcs,
            optimizer,
            scheduler,
            max_epochs,
            config,
        )
        self.config = config
        self.device = device

        self.data = data
        self.validation_data = validation_data

        self.do_validation = True

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()
        self.batch_log.reset()

        for batch_idx, (data, target) in enumerate(self.data):
            input, target = (
                data.to(self.device),
                target.to(self.device),
            )

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            output = self.model(input)

            # Compute the loss and its gradients
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping: Scale gradients if they exceed max_norm
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            # Adjust learning weights
            self.optimizer.step()

            # Log the results
            self.batch_log.update("batch", batch_idx)
            self.batch_log.update("loss", loss.item())
            for met in self.metric_funcs:
                self.batch_log.update(met.__name__, met(output, target))
        
        # Run validation
        if self.do_validation:
            val_loss = self._validation_epoch(epoch)
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: Learning Rate = {current_lr}")

    def _validation_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        with torch.no_grad():

            for batch_idx, (data, target) in enumerate(self.validation_data):
                input, target = (
                    data.to(self.device),
                    target.to(self.device),
                )

                output = self.model(input)
                loss = self.criterion(output, target)

                # Log the results
                self.batch_log.update("val_loss", loss.item())
                for met in self.metric_funcs:
                    self.batch_log.update("val_" + met.__name__, met(output, target))
        return loss
