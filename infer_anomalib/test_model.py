import sys
from PIL import Image
from anomalib.data.utils import ValSplitMode

sys.path.append('')

from anomalib.data.image.folder import Folder
from anomalib import TaskType

from anomalib.deploy import TorchInferencer
import numpy as np
import cv2
from torch import as_tensor
from torchvision.transforms.v2.functional import to_dtype, to_image
import torch
import argparse
from utils import show_image_list
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os


class2label = {"NORMAL": 0, "ABNORMAL": 1}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_torch_model', type=str, help='Path to the torch model')
    parser.add_argument('--path_dataset', type=str, help='Path to the image to analyze')
    parser.add_argument('--name', type=str, help='Name the current dataset')
    parser.add_argument('--dir_result', type=str, help='Directory where to store the results')
    parser.add_argument('--image_size', type=int, default=256, help='Size of the image')
    parser.add_argument('--normal_dir', type=str, default="90_DEG", help='Name of the normal directory')

    opt = parser.parse_args()

    # load config file
    path_torch_model = opt.path_torch_model
    path_dataset = opt.path_dataset
    dir_result = opt.dir_result
    name = opt.name
    image_size = int(opt.image_size)
    normal_dir = opt.normal_dir

    # load the model
    inferencer = TorchInferencer(path=path_torch_model,
                                 device="cpu")

    # load the datamodule
    datamodule = Folder(
        name=name,
        root=path_dataset,
        normal_dir=normal_dir,
        abnormal_dir="abnormal",
        task=TaskType.CLASSIFICATION,
        seed=42,
        val_split_mode=ValSplitMode.FROM_TEST,  # default value
        val_split_ratio=0.5,  # default value
        #image_size=(image_size,image_size)
    )

    # Setup the datamodule
    datamodule.setup()

    # take the test dataset
    test_dataset = datamodule.test_data.samples

    # create the dataset to store the results
    df_results = pd.DataFrame(columns=['IMG_PATH', 'TRUE_CATEGORY', 'TRUE_LABEL', 'PRED_CATEGORY', 'PRED_LABEL', 'PRED_SCORE', 'DEG'])

    # list to collect the true and pred labels
    true_label_list = []
    pred_label_list = []

    # iter over the test dataset
    for index, row in test_dataset.iterrows():
        image_path = row["image_path"]
        true_label = row["label_index"]

        deg = image_path.split("/")[-2]

        image = Image.open(image_path).convert("RGB")
        image = image.resize((image_size, image_size))
        image = to_dtype(to_image(image), torch.float32, scale=True) if as_tensor else np.array(image) / 255.0

        result = inferencer.predict(image=image)

        true_category = ""
        if true_label == 0:
            true_category = "NORMAL"
        elif true_label == 1:
            true_category = "ABNORMAL"

        pred_category = ""
        pred_score = 0
        if result.pred_label == 0:
            pred_score = 1 - result.pred_score
            pred_category = "NORMAL"
        elif result.pred_label == 1:
            pred_score = result.pred_score
            pred_category = "ABNORMAL"

        true_label_list.append(true_label)
        pred_label_list.append(result.pred_label)

        df_results = df_results._append({'IMG_PATH': image_path,
                                        'TRUE_CATEGORY': true_category,
                                        'TRUE_LABEL': true_label,
                                        'PRED_CATEGORY': pred_category,
                                        'PRED_LABEL': result.pred_label,
                                        'PRED_SCORE': pred_score,
                                        'DEG': deg}, ignore_index=True)


    print('Accuracy: ', accuracy_score(true_label_list, pred_label_list))
    print(metrics.classification_report(true_label_list, pred_label_list))

    ## Plot and save the confusion matrix
    cm = metrics.confusion_matrix(true_label_list, pred_label_list)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.set(font_scale=1.3)  # Adjust to fit
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=class2label.keys(),
           yticklabels=class2label.keys())
    # plt.yticks(fontsize=10, rotation=0)
    plt.yticks(fontsize=11, rotation=-30, ha='right', rotation_mode='anchor')
    # plt.xticks(fontsize=10, rotation=90)
    plt.xticks(fontsize=11, rotation=30, ha='right', rotation_mode='anchor')

    fig.savefig(os.path.join(dir_result, name + "_confusion_matrix.png"))


    ## Check distribution score over the right classification
    df_results_correct = df_results[df_results["TRUE_CATEGORY"] == df_results["PRED_CATEGORY"]]
    # boxplot
    plt.figure(figsize=(15, 15))
    sns.boxplot(data=df_results_correct, x="DEG")
    plt.xticks(rotation=45)
    plt.title('DEG distribution classification', fontsize=12)
    plt.savefig(os.path.join(dir_result, "plot_bad_classification.png"))


    ## Check deg distribution over the error classification
    df_results_error = df_results[df_results["TRUE_CATEGORY"] != df_results["PRED_CATEGORY"]]
    df2 = df_results_error.groupby(['DEG']).size().reset_index(name='COUNT')
    plt.figure(figsize=(15, 15))
    # create grouped bar chart
    barplot = sns.barplot(x='DEG', y='COUNT', data=df2, orient='v').set(title='Number of error per DEG')
    plt.savefig(os.path.join(dir_result, "plot_bad_classification.png"))

    # save the csv with results
    df_results.to_csv(os.path.join(dir_result, name + "_result.csv"), index=False)