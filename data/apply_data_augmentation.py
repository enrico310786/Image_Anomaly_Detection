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

# show images
def img_is_color(img):
    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False


def show_image_list(list_images, name_image, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(20, 10),
                    title_fontsize=30):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    '''

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images = len(list_images)
    num_cols = min(num_images, num_cols)
    num_rows = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):
        img = list_images[i]
        title = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')

        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize)
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    plt.savefig("/home/enrico/Projects/Image_Anomaly_Detection/data/augment_" + name_image + ".png")
    _ = plt.show()


show_image_list(list_images=[np.array(img_train_0), np.array(img_train_1), np.array(img_train_2), np.array(img_train_3), np.array(img_train_4), np.array(img_train_5)],
                list_titles=['img_0', 'img_1', 'img_2', 'img_3', 'img_4','img_5'],
                num_cols=3,
                figsize=(20, 10),
                grid=False,
                title_fontsize=20,
                name_image="train_image_examples")

show_image_list(list_images=[np.array(img_val_0), np.array(img_val_1), np.array(img_val_2), np.array(img_val_3), np.array(img_val_4), np.array(img_val_5)],
                list_titles=['img_0', 'img_1', 'img_2', 'img_3', 'img_4','img_5'],
                num_cols=3,
                figsize=(20, 10),
                grid=False,
                title_fontsize=20,
                name_image="val_image_examples")