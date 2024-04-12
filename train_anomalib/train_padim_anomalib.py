from anomalib.data.utils import ValSplitMode
from anomalib.deploy import ExportType
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from anomalib.models import Padim
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
    parser.add_argument('--result_directory', type=str)
    parser.add_argument('--name_wandb_experiment', type=str)
    opt = parser.parse_args()

    dataset_root = opt.dataset_root
    result_directory = opt.result_directory
    name_wandb_experiment = opt.name_wandb_experiment

    datamodule = Folder(
        name="one_up",
        root=dataset_root,
        normal_dir="one_up",
        abnormal_dir="abnormal",
        #transform=transform,
        task=TaskType.CLASSIFICATION,
        seed=42,
        val_split_mode=ValSplitMode.FROM_TEST, # default value
        val_split_ratio=0.5, # default value
        image_size=(256,256)
    )

    model = Padim()

    callbacks = [
        ModelCheckpoint(
            dirpath=result_directory,
            mode="max",
            monitor="image_AUROC",
        ),
        EarlyStopping(
            monitor="image_AUROC",
            mode="max",
            patience=3,
        ),
    ]

    wandb_logger = AnomalibWandbLogger(project="image_anomaly_detection",
                                       name=name_wandb_experiment)

    engine = Engine(
        callbacks=callbacks,
        pixel_metrics="AUROC",
        accelerator="auto",  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
        devices=1,
        logger=wandb_logger,
        task=TaskType.CLASSIFICATION,
    )

    engine.fit(datamodule=datamodule, model=model)

    engine.test(datamodule=datamodule, model=model)

    #export in torch
    path_export_weights = engine.export(export_type=ExportType.TORCH,
                                        model=model,
                                        export_root=os.path.join(result_directory, "weights"))

    print("path_export_weights: ", path_export_weights)