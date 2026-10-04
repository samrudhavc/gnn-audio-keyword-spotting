import yaml
import dotmap
import lightning as L
import torch
import argparse
import multiprocessing as mp

from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from models.recognition_lipsfus import LNRecognition
from data.tdigits import TDIGITS

def main():
    cfg = yaml.load(open('configs/tdigits.yaml', 'r'), Loader=yaml.FullLoader)
    cfg = dotmap.DotMap(cfg)

    dm = TDIGITS(cfg)
    dm.setup()

    model = LNRecognition(cfg)

    wandb_logger = WandbLogger(project='audio_event_tdigits', name=f'tdigits')
    wandb_logger.watch(model)

    lr_monitor = LearningRateMonitor(logging_interval='step')


    print(cfg)
    print(model.model)

    print("\n#####################################################################################")
    print("############################### TRAINING FLOAT MODEL ################################")
    print("#####################################################################################\n")

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best_model_float',
        monitor='val_acc',
        mode='max',
        save_top_k=1
    )

    trainer = L.Trainer(max_epochs=100, 
                        log_every_n_steps=1, 
                        gradient_clip_val=0.0,
                        logger=wandb_logger,
                        callbacks=[lr_monitor, checkpoint_callback],
                        deterministic=True)


    trainer.fit(model, dm)
    best_model_path = checkpoint_callback.best_model_path
    print(f"\nBest float model saved at: {best_model_path}\n")
    model = LNRecognition.load_from_checkpoint(best_model_path)
    trainer.test(model, datamodule=dm)


    print(" \n#####################################################################################")
    print("############################### TRAINING QAT MODEL ################################")
    print("#####################################################################################\n")

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best_model_calibrated',
        monitor='val_acc',
        mode='max',
        save_top_k=1
    )

    trainer = L.Trainer(max_epochs=20,
                        log_every_n_steps=1, 
                        gradient_clip_val=0.0,
                        logger=wandb_logger,
                        callbacks=[lr_monitor, checkpoint_callback],
                        deterministic=True)
    
    model.model.calibrate()
    trainer.fit(model, dm)
    best_model_path = checkpoint_callback.best_model_path
    print(f"\nBest model saved at: {best_model_path}\n")
    model = LNRecognition.load_from_checkpoint(best_model_path)

    trainer.test(model, datamodule=dm)

    print(" \n#####################################################################################")
    print("############################### QUANTIZING MODEL ################################")
    print("#####################################################################################\n")

    model.model.quantize()
    trainer.test(model, datamodule=dm)
    
    torch.save(model.model, best_model_path.replace('calibrated', 'quantized'))


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    L.seed_everything(42)
    main()