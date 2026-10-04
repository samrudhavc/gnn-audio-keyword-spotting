import torch
import lightning as L


from torchmetrics.functional.classification import accuracy
from torchmetrics import Accuracy
from torchmetrics.classification import ConfusionMatrix

from typing import Dict, Tuple
from torch.nn.functional import softmax

from models.networks.real_model import GCN

import wandb
import numpy as np
import matplotlib.pyplot as plt



class LNRecognition(L.LightningModule):
    def __init__(self, 
                 config):
        super().__init__()

        self.config = config

        self.lr = config.train.lr
        self.weight_decay = config.train.weight_decay

        self.batch_size = config.train.batch_size
        self.num_classes = config.model.num_classes

        self.model = GCN(config)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()

        self.conf_matrix = ConfusionMatrix(num_classes=self.num_classes, normalize='true', task="multiclass")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.lr, 
                                     weight_decay=self.weight_decay)

        if self.config.train.use_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                    mode='max',
                                                                    factor=0.5,
                                                                    patience=5,
                                                                    verbose=True)
            print(optimizer)
            print(lr_scheduler)
            return {'optimizer': optimizer, 
                    'lr_scheduler': lr_scheduler,
                    'monitor': 'val_acc'}
        print(optimizer)
        print("No scheduler used")
        return optimizer

    def forward(self, data):
        x, _ = self.model(data)
        return x

    def training_step(self, batch, batch_idx):
        outputs = self.forward(data=batch)
        loss = self.criterion(outputs, target=batch['y'])

        y_prediction = torch.argmax(outputs, dim=-1)
        acc = accuracy(preds=y_prediction, target=batch['y'], task="multiclass", num_classes=self.num_classes)

        self.log('train_loss', loss, on_epoch=True, logger=True, batch_size=self.batch_size, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, logger=True, batch_size=self.batch_size, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(data=batch)
        
        loss = self.criterion(outputs, target=batch['y'])
        y_prediction = torch.argmax(outputs, dim=-1)

        acc = accuracy(preds=y_prediction, target=batch['y'], task="multiclass", num_classes=self.num_classes)
        self.log('val_loss', loss, on_epoch=True, logger=True, batch_size=self.batch_size, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, logger=True, batch_size=self.batch_size, prog_bar=True)

        # add confusion matrix
        self.conf_matrix.update(preds=y_prediction, target=batch['y'])

    def on_validation_epoch_end(self):
        conf_matrix = self.conf_matrix.compute()
        self.conf_matrix.reset()

        print(conf_matrix)
        # # make a heatmap of the confusion matrix
        # conf_matrix_np = conf_matrix.cpu().numpy()
        # plt.figure(figsize=(10, 10))
        # plt.imshow(conf_matrix_np, interpolation='nearest', cmap=plt.cm.Blues)
        # plt.title('Confusion Matrix')
        # plt.colorbar()
        # tick_marks = np.arange(self.num_classes)
        # plt.xticks(tick_marks, range(self.num_classes), rotation=45)
        # plt.yticks(tick_marks, range(self.num_classes))
        # plt.xlabel('Predicted label')
        # plt.ylabel('True label')
        # plt.savefig('confusion_matrix.png')


    def test_step(self, batch, batch_idx):
        outputs = self.forward(data=batch)
        loss = self.criterion(outputs, target=batch['y'])
        y_prediction = torch.argmax(outputs, dim=-1)
        
        acc = accuracy(preds=y_prediction, target=batch['y'], task="multiclass", num_classes=self.num_classes)

        self.log('test_loss', loss, on_epoch=True, logger=True, batch_size=self.batch_size, prog_bar=True)
        self.log('test_acc', acc, on_epoch=True, logger=True, batch_size=self.batch_size, prog_bar=True)