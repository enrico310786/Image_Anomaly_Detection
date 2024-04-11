from anomalib.data.utils import ValSplitMode
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from anomalib.models import Fastflow
from anomalib import TaskType
from anomalib.data.image.folder import Folder
from anomalib.loggers import AnomalibWandbLogger
from anomalib.engine import Engine

# follow the notebook https://github.com/openvinotoolkit/anomalib/blob/main/notebooks/200_models/201_fastflow.ipynb

# set the dataset root for a particular category
dataset_root = "/home/enrico/Dataset/images_anomaly/dataset_lego/images_lego/one_up"

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

model = Fastflow(backbone="resnet18", flow_steps=8)

callbacks = [
    ModelCheckpoint(
        dirpath="results/fastflow/",
        mode="max",
        monitor="image_AUROC",
    ),
    EarlyStopping(
        monitor="image_AUROC",
        mode="max",
        patience=3,
    ),
]


wandb_logger = AnomalibWandbLogger(project="image_anomaly_detection", name="fastflow_1")
engine = Engine(
    callbacks=callbacks,
    pixel_metrics="AUROC",
    accelerator="auto",  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
    devices=1,
    logger=wandb_logger,
)

engine.fit(datamodule=datamodule, model=model)

engine.test(datamodule=datamodule, model=model)