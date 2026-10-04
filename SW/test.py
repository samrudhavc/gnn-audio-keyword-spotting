import yaml
import dotmap
import lightning as L
import argparse
import multiprocessing as mp
import torch

from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from models.recognition import LNRecognition
from data.spiking_digits import SpikingDigits

best_model_path = "checkpoints/best_model-v2.ckpt"
print(f"Best model saved at: {best_model_path}")

cfg = yaml.load(open('configs/digits.yaml', 'r'), Loader=yaml.FullLoader)
cfg = dotmap.DotMap(cfg)

dm = SpikingDigits(cfg)
dm.setup()

model = LNRecognition(cfg)

model.model.freeze()
model.model.load_state_dict(torch.load('model.pth'))


print(model.model.state_dict())

trainer = L.Trainer(max_epochs=1, 
                        log_every_n_steps=1, 
                        gradient_clip_val=0.0,
                        # logger=wandb_logger,
                        # callbacks=[lr_monitor, checkpoint_callback],
                        deterministic=True)

trainer.test(model, datamodule=dm)