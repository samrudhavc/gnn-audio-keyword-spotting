import os
import yaml
import dotmap
import lightning as L
import torch
import multiprocessing as mp

#from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from models.recognition import LNRecognition
from data.spiking_digits import SpikingDigits


def make_trainer(logger, callbacks, max_epochs):
    return L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=max_epochs,
        precision="32-true",
        log_every_n_steps=10,
        gradient_clip_val=0.0,
        logger=logger,
        callbacks=callbacks,
        deterministic=False,
        benchmark=True,
    )


def main():

    # --------------------------------------------------
    # GPU CHECK
    # --------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. Check the Docker GPU configuration."
        )

    print("\nGPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    print("PyTorch version:", torch.__version__)

    torch.set_float32_matmul_precision("high")

    # --------------------------------------------------
    # CONFIGURATION AND DATA
    # --------------------------------------------------

    with open("configs/digits.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = dotmap.DotMap(cfg)

    dm = SpikingDigits(cfg)
    dm.setup()

    # --------------------------------------------------
    # LOGGING
    # --------------------------------------------------

    local_logger = CSVLogger(
        save_dir="training_logs",
        name="shd_float_gpu"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ==================================================
    # 1. FLOATING-POINT MODEL TRAINING
    # ==================================================

    print("\n" + "#" * 80)
    print("TRAINING FLOATING-POINT MODEL ON GPU")
    print("#" * 80)

    float_model = LNRecognition(cfg)

    print(float_model.model)

    float_checkpoint = ModelCheckpoint(
        dirpath="checkpoints/float_gpu",
        filename="best_model_float_gpu-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    float_trainer = make_trainer(
        logger=local_logger,
        callbacks=[lr_monitor, float_checkpoint],
        max_epochs=100,
    )

    # Train from scratch
    float_trainer.fit(float_model, datamodule=dm)

    best_float_path = float_checkpoint.best_model_path

    if not best_float_path:
        raise RuntimeError("No best floating-point checkpoint was saved.")

    print("\nBest float checkpoint:", best_float_path)

    # Evaluate best floating-point checkpoint
    float_model = LNRecognition.load_from_checkpoint(
        best_float_path,
        weights_only=False,
    )

    float_trainer.test(float_model, datamodule=dm)

    # ==================================================
    # 2. CALIBRATION / QAT MODEL TRAINING
    # ==================================================

    print("\n" + "#" * 80)
    print("TRAINING CALIBRATED / QAT MODEL ON GPU")
    print("#" * 80)

    # Start from the newly trained best float model
    qat_model = LNRecognition.load_from_checkpoint(
        best_float_path,
        weights_only=False,
    )

    # Apply the repository's calibration procedure
    qat_model.model.calibrate()

    qat_checkpoint = ModelCheckpoint(
        dirpath="checkpoints/calibrated_gpu",
        filename="best_model_calibrated_gpu-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    qat_trainer = make_trainer(
        #logger=wandb_logger,
        logger=local_logger,
        callbacks=[lr_monitor, qat_checkpoint],
        max_epochs=20,
    )

    # Train the calibrated/QAT model
    qat_trainer.fit(qat_model, datamodule=dm)

    best_qat_path = qat_checkpoint.best_model_path

    if not best_qat_path:
        raise RuntimeError("No best calibrated checkpoint was saved.")

    print("\nBest calibrated checkpoint:", best_qat_path)

    # Evaluate the best calibrated checkpoint
    qat_model = LNRecognition.load_from_checkpoint(
        best_qat_path,
        weights_only=False,
    )

    qat_trainer.test(qat_model, datamodule=dm)

    # ==================================================
    # 3. QUANTIZED MODEL EVALUATION
    # ==================================================

    print("\n" + "#" * 80)
    print("QUANTIZED MODEL EVALUATION")
    print("#" * 80)

    qat_model.model.quantize()

    qat_trainer.test(qat_model, datamodule=dm)

    print("\nTraining and evaluation completed.")


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    L.seed_everything(42, workers=True)

    main()
