import numpy as np
from anomalib.data.utils import ValSplitMode
from torchvision.transforms.v2.functional import to_pil_image
from anomalib.data.image.folder import Folder
from anomalib import TaskType
import os
import torch

from utils import show_image_list

# set the dataset root for a particular category
dataset_root = "/home/enrico/Projects/Image_Anomaly_Detection/dataset/images_lego_256/one_up"


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
    normal_split_ratio=0.2, # default value
    val_split_mode=ValSplitMode.FROM_TEST, # default value
    val_split_ratio=0.5, # default value
    train_batch_size=32, # default value
    eval_batch_size=32, # default value
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
print("data_val['image_path'][0]: {} - data_val['image'][0].shape: {} - data_val['label'][0]: {}".format(data_val['image_path'][0], data_val['image'][0].shape, data_val['label'][0]))
img_val = to_pil_image(data_val["image"][0].clone())


# test images
i, data_test = next(enumerate(datamodule.test_dataloader()))
# for each key extract the first image
print("data_test['image_path'][0]: {} - data_test['image'][0].shape: {} - data_test['label'][0]: {}".format(data_test['image_path'][0], data_test['image'][0].shape, data_test['label'][0]))
img_test = to_pil_image(data_test["image"][0].clone())


# from the datamodule extract the train, val and test Pandas dataset and collect all the info in a csv
train_dataset = datamodule.train_data.samples
print("train_dataset.head()")
print(train_dataset[['label','label_index']].head())
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

show_image_list(list_images=[np.array(img_train), np.array(img_val), np.array(img_test)],
                list_titles=['train', 'validation', 'test'],
                num_cols=3,
                figsize=(20, 10),
                grid=False,
                title_fontsize=20,
                path_image="/home/enrico/Projects/Image_Anomaly_Detection/resources/examples_images.png")