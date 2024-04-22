from anomalib.data.utils import ValSplitMode
from anomalib.deploy import ExportType
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from anomalib.models import EfficientAd
from anomalib import TaskType
from anomalib.data.image.folder import Folder
from anomalib.loggers import AnomalibWandbLogger
from anomalib.engine import Engine
import argparse
from torchvision.transforms import v2


# follow the notebook https://github.com/openvinotoolkit/anomalib/blob/main/notebooks/200_models/201_fastflow.ipynb

'''
Set task=TaskType.CLASSIFICATION in engine
Set the export to torch for inference
'''

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, )
    parser.add_argument('--name_normal_dir', type=str)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--name_wandb_experiment', type=str)
    parser.add_argument("--data_augmentation", type=str2bool, nargs='?', const=True, default=False, help="Apply data augmentation during train")

    opt = parser.parse_args()

    dataset_root = opt.dataset_root
    name_wandb_experiment = opt.name_wandb_experiment
    name_normal_dir = opt.name_normal_dir
    max_epochs = int(opt.max_epochs)
    data_augmentation = opt.data_augmentation
    patience = opt.patience

    # Define a list of transformations you want to apply to your data
    transformations_list = [# geometric transformations
                            v2.RandomPerspective(distortion_scale=0.2, p=0.5),
                            v2.RandomAffine(degrees=(0, 1), scale=(0.9, 1.2)),
                            # pixel transformations
                            v2.ColorJitter(brightness=(0.6, 1.6)),
                            v2.ColorJitter(contrast=(0.6, 1.6)),
                            v2.ColorJitter(saturation=(0.6, 1.6)),
                            v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.3, 0.3))
                            ]

    transforms = None
    if data_augmentation:
        print("Apply data augmentation")
        transforms = v2.RandomApply(transformations_list, p=0.8)

    datamodule = Folder(
        name="one_up",
        root=dataset_root,
        normal_dir=name_normal_dir,
        abnormal_dir="abnormal",
        task=TaskType.CLASSIFICATION,
        seed=42,
        val_split_mode=ValSplitMode.FROM_TEST, # default value
        val_split_ratio=0.5, # default value
        train_transform=transforms
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
            patience=patience,
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

    print("Fit...")
    engine.fit(datamodule=datamodule, model=model)

    print("Test...")
    engine.test(datamodule=datamodule, model=model)

    print("Export weights...")
    path_export_weights = engine.export(export_type=ExportType.TORCH,
                                        model=model)

    print("path_export_weights: ", path_export_weights)