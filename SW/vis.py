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
import cv2
import matplotlib.pyplot as plt

def main():
    cfg = yaml.load(open('configs/tdigits.yaml', 'r'), Loader=yaml.FullLoader)
    cfg = dotmap.DotMap(cfg)
    cfg.train.batch_size = 1  # Set batch size to 64
    cfg.train.num_workers = 0  # Set num_workers to 0 for debugging

    dm = TDIGITS(cfg)
    dm.setup()

    for data in dm.val_dataloader():
        print(data)
        print(data['y'])
        pos = data['pos'].cpu().numpy()
        edges = data['edge_index'].cpu().numpy()

        # visualize the data

        plt.figure(figsize=(10, 10))
        plt.scatter(pos[:, 0], pos[:, 1], c='blue', s=1, label='Lips Points')
        # for edge in edges.T:
        #     plt.plot(pos[edge, 0], pos[edge, 1], c='red', linewidth=0.5, alpha=0.5)
        # save in high resolution
        plt.title('Lips Points Visualization')
        # plt.savefig('lips_points.png', dpi=300, bbox_inches='tight')
        plt.show()

        # break



if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    L.seed_everything(42)
    main()