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

# set the dataset root for a particular category
dataset_root = "/home/enrico/Dataset/images_anomaly/dataset_lego/images_lego_256/one_up"

# set the transformation
image_size = (256, 256)
transform = Resize(image_size, antialias=True)

'''
val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
    Defaults to ``ValSplitMode.FROM_TEST``.
val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
    Defaults to ``0.5``.
    
By default the validation set is created from the test set -> this is controlled by val_split_mode=ValSplitMode.FROM_TEST
The fraction of test data used to build the validation data is set by the parameter val_split_ratio uqual to 0.5 by dedfault
'''
# Create the datamodule
datamodule = Folder(
    name="one_up",
    root=dataset_root,
    normal_dir="90_DEG",
    abnormal_dir="abnormal",
    task=TaskType.CLASSIFICATION,
    seed=42,
    val_split_mode=ValSplitMode.FROM_TEST, # default value
    val_split_ratio=0.5, # default value
    #image_size=(512,512)
)

# Setup the datamodule
datamodule.setup()

'''
https://anomalib.readthedocs.io/en/v1.0.1/markdown/guides/how_to/data/custom_data.html

The Folder datamodule will create training, validation, test and prediction datasets and dataloaders for us. 
We can access the datasets and dataloaders by following the same approach as in the segmentation task.

When we check the samples from the dataloaders, we will see that the mask key is not present in the samples. 
This is because we do not need the masks for the classification task.

'''

# Train images
i, data_train = next(enumerate(datamodule.train_dataloader()))
print(data_train.keys(), data_train["image"].shape) # it takes a batch of images
# for each key extract the first image
print("data_train['image_path'][0]: {} - data_train['image'][0].shape: {} - data_train['label'][0]: {} - torch.max(data_train['image][0]): {} - torch.min(data_train['image][0]): {}".format(data_train['image_path'][0], data_train['image'][0].shape, data_train['label'][0], torch.max(data_train['image'][0]), torch.min(data_train['image'][0])))
img_train = to_pil_image(data_train["image"][0].clone())

# val images
i, data_val = next(enumerate(datamodule.val_dataloader()))
# for each key extract the first image
print("data_val['image_path'][100]: {} - data_val['image'][100].shape: {} - data_val['label'][100]: {}".format(data_val['image_path'][10], data_val['image'][10].shape, data_val['label'][10]))
img_val = to_pil_image(data_val["image"][1].clone())


# test images
i, data_test = next(enumerate(datamodule.test_dataloader()))
# for each key extract the first image
print("data_test['image_path'][100]: {} - data_test['image'][100].shape: {} - data_test['label'][100]: {}".format(data_test['image_path'][10], data_test['image'][10].shape, data_test['label'][10]))
img_test = to_pil_image(data_test["image"][0].clone())


# from the datamodule extract the train, val and test Pandas dataset and collect all the info in a csv
train_dataset = datamodule.train_data.samples
test_dataset = datamodule.test_data.samples
val_dataset = datamodule.val_data.samples

# check the data distribution for each category in each data split
print("TRAIN DATASET FEATURES")
print(train_dataset.info())
print("")
print("IMAGE DISTRIBUTION BY CLASS")
print("")
desc_grouped = train_dataset[['label']].value_counts()
print(desc_grouped)
print("----------------------------------------------------------")
print("TEST DATASET FEATURES")
print(test_dataset.info())
print("")
print("IMAGE DISTRIBUTION BY CLASS")
print("")
desc_grouped = test_dataset[['label']].value_counts()
print(desc_grouped)
print("----------------------------------------------------------")
print("VAL DATASET FEATURES")
print(val_dataset.info())
print("")
print("IMAGE DISTRIBUTION BY CLASS")
print("")
desc_grouped = val_dataset[['label']].value_counts()
print(desc_grouped)

datamodule.train_data.samples.to_csv(os.path.join("/home/enrico/Projects/Image_Anomaly_Detection/data", "datamodule_train.csv"), index=False)
datamodule.test_data.samples.to_csv(os.path.join("/home/enrico/Projects/Image_Anomaly_Detection/data", "datamodule_test.csv"), index=False)
datamodule.val_data.samples.to_csv(os.path.join("/home/enrico/Projects/Image_Anomaly_Detection/data", "datamodule_val.csv"), index=False)


# show images
def img_is_color(img):

    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False

def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(20, 10),
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
    plt.savefig('/home/enrico/Projects/Image_Anomaly_Detection/data/examples_images.png')
    _ = plt.show()



show_image_list(list_images=[np.array(img_train), np.array(img_val), np.array(img_test)],
                list_titles=['train', 'validation', 'test'],
                num_cols=3,
                figsize=(20, 10),
                grid=False,
                title_fontsize=20)