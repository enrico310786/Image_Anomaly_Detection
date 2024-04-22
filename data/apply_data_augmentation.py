import numpy as np
from PIL import Image
from anomalib.data.utils import ValSplitMode
from torchvision.transforms.v2 import Resize
from torchvision.transforms.v2.functional import to_pil_image
import matplotlib.pyplot as plt
from anomalib.data.image.folder import Folder
from anomalib import TaskType
import os
import torch
from torchvision.transforms import v2

from utils import show_image_list

# set the dataset root for a particular category
dataset_root = "/home/enrico/Dataset/images_anomaly/dataset_lego/images_lego_256/one_up"

'''
transforms = v2.Compose([
    # geometric transformations
    v2.RandomPerspective(distortion_scale=0.2, p=0.5),
    v2.RandomAffine(degrees=(0, 1), scale=(0.9, 1.2)),
    # pixel transformations
    v2.ColorJitter(brightness=(0.6, 1.6)),
    v2.ColorJitter(contrast=(0.6, 1.6)),
    v2.ColorJitter(saturation=(0.6, 1.6)),
    v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.3, 0.3)),
])
'''

# Define a list of transformations you want to apply to your data
transformations_list = [  # geometric transformations
    v2.RandomPerspective(distortion_scale=0.2, p=0.5),
    v2.RandomAffine(degrees=(0, 1), scale=(0.9, 1.2)),
    # pixel transformations
    v2.ColorJitter(brightness=(0.6, 1.6)),
    v2.ColorJitter(contrast=(0.6, 1.6)),
    v2.ColorJitter(saturation=(0.6, 1.6)),
    v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.3, 0.3))
]


transforms = v2.RandomApply(transformations_list, p=0.7)


# Create the datamodule
datamodule = Folder(
    name="one_up",
    root=dataset_root,
    normal_dir="90_DEG",
    abnormal_dir="abnormal",
    task=TaskType.CLASSIFICATION,
    seed=42,
    val_split_mode=ValSplitMode.FROM_TEST,  # default value
    val_split_ratio=0.5,  # default value
    train_transform=transforms
)

# Setup the datamodule
datamodule.setup()


# Train images
i, data_train = next(enumerate(datamodule.train_dataloader()))
print(data_train.keys(), data_train["image"].shape)  # it takes a batch of images
# for each key extract the first image
print(
    "data_train['image_path'][0]: {} - data_train['image'][0].shape: {} - data_train['label'][0]: {} - torch.max(data_train['image][0]): {} - torch.min(data_train['image][0]): {}".format(
        data_train['image_path'][0], data_train['image'][0].shape, data_train['label'][0],
        torch.max(data_train['image'][0]), torch.min(data_train['image'][0])))
img_train_0 = to_pil_image(data_train["image"][0].clone())
img_train_1 = to_pil_image(data_train["image"][1].clone())
img_train_2 = to_pil_image(data_train["image"][2].clone())
img_train_3 = to_pil_image(data_train["image"][3].clone())
img_train_4 = to_pil_image(data_train["image"][4].clone())
img_train_5 = to_pil_image(data_train["image"][5].clone())


# Validation images
i, data_val = next(enumerate(datamodule.val_dataloader()))
print(data_val.keys(), data_val["image"].shape)  # it takes a batch of images
# for each key extract the first image
print(
    "data_val['image_path'][0]: {} - data_val['image'][0].shape: {} - data_val['label'][0]: {} - torch.max(data_val['image][0]): {} - torch.min(data_val['image][0]): {}".format(
        data_val['image_path'][0], data_val['image'][0].shape, data_val['label'][0],
        torch.max(data_val['image'][0]), torch.min(data_val['image'][0])))
img_val_0 = to_pil_image(data_val["image"][0].clone())
img_val_1 = to_pil_image(data_val["image"][1].clone())
img_val_2 = to_pil_image(data_val["image"][2].clone())
img_val_3 = to_pil_image(data_val["image"][3].clone())
img_val_4 = to_pil_image(data_val["image"][4].clone())
img_val_5 = to_pil_image(data_val["image"][5].clone())


show_image_list(list_images=[np.array(img_train_0), np.array(img_train_1), np.array(img_train_2), np.array(img_train_3), np.array(img_train_4), np.array(img_train_5)],
                list_titles=['img_0', 'img_1', 'img_2', 'img_3', 'img_4','img_5'],
                num_cols=3,
                figsize=(20, 10),
                grid=False,
                title_fontsize=20,
                path_image="/home/enrico/Projects/Image_Anomaly_Detection/data/augment_train_image_examples.png")

show_image_list(list_images=[np.array(img_val_0), np.array(img_val_1), np.array(img_val_2), np.array(img_val_3), np.array(img_val_4), np.array(img_val_5)],
                list_titles=['img_0', 'img_1', 'img_2', 'img_3', 'img_4','img_5'],
                num_cols=3,
                figsize=(20, 10),
                grid=False,
                title_fontsize=20,
                path_image="/home/enrico/Projects/Image_Anomaly_Detection/data/augment_val_image_examples.png")