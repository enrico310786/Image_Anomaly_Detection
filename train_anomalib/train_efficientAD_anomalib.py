from anomalib.data.utils import ValSplitMode
from anomalib.deploy import ExportType
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from anomalib.models import EfficientAd
from anomalib import TaskType
from anomalib.data.image.folder import Folder
from anomalib.loggers import AnomalibWandbLogger
from anomalib.engine import Engine
import argparse
import os

# follow the notebook https://github.com/openvinotoolkit/anomalib/blob/main/notebooks/200_models/201_fastflow.ipynb

'''
Set task=TaskType.CLASSIFICATION in engine
Set the export to torch for inference
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, )
    parser.add_argument('--name_normal_dir', type=str)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--name_wandb_experiment', type=str)

    opt = parser.parse_args()

    dataset_root = opt.dataset_root
    name_wandb_experiment = opt.name_wandb_experiment
    name_normal_dir = opt.name_normal_dir
    max_epochs = int(opt.max_epochs)

    datamodule = Folder(
        name="one_up",
        root=dataset_root,
        normal_dir=name_normal_dir,
        abnormal_dir="abnormal",
        #transform=transform,
        task=TaskType.CLASSIFICATION,
        seed=42,
        val_split_mode=ValSplitMode.FROM_TEST, # default value
        val_split_ratio=0.5, # default value
        image_size=(256,256)
    )

    model = EfficientAd()

    callbacks = [
        ModelCheckpoint(
            mode="max",
            monitor="image_AUROC",
            save_last=True,
            verbose=True,
            auto_insert_metric_name=True,
            every_n_epochs=1,
        ),
        EarlyStopping(
            monitor="image_AUROC",
            mode="max",
            patience=5,
        ),
    ]

    wandb_logger = AnomalibWandbLogger(project="image_anomaly_detection",
                                       name=name_wandb_experiment)

    engine = Engine(
        max_epochs=max_epochs,
        callbacks=callbacks,
        pixel_metrics="AUROC",
        accelerator="auto",  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
        devices=1,
        logger=wandb_logger,
        task=TaskType.CLASSIFICATION,
    )

    engine.fit(datamodule=datamodule, model=model)

    engine.test(datamodule=datamodule, model=model)

    path_export_weights = engine.export(export_type=ExportType.TORCH,
                                        model=model)

    print("path_export_weights: ", path_export_weights)