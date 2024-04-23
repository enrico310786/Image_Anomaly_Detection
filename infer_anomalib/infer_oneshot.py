import sys
from PIL import Image
sys.path.append('')

from anomalib.deploy import TorchInferencer
import numpy as np
import cv2
from torch import as_tensor
from torchvision.transforms.v2.functional import to_dtype, to_image
import torch
import argparse
from utils import show_image_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_torch_model', type=str, help='Path to the torch model')
    parser.add_argument('--path_image', type=str, help='Path to the image to analyze')
    parser.add_argument('--path_result', type=str, help='Path where to save the result image')
    opt = parser.parse_args()

    # 1 - load config file
    path_torch_model = opt.path_torch_model
    path_image = opt.path_image
    path_result = opt.path_result

    '''
    """Apply min-max normalization and shift the values such that the threshold value is centered at 0.5."""
    '''
    inferencer = TorchInferencer(path=path_torch_model,
                                 device="cpu")

    image = Image.open(path_image).convert("RGB")
    image = image.resize((256, 256))
    image = to_dtype(to_image(image), torch.float32, scale=True) if as_tensor else np.array(image) / 255.0

    result = inferencer.predict(image=image)

    print("inferencer.metadata", inferencer.metadata)
    print("pred_score: {:.4f} - pred_label: {}".format(result.pred_score, result.pred_label))

    if result.pred_label == 0:
        normal_score = 1 - result.pred_score
        print("Normal - pred_score: {:.4f}".format(normal_score))
    else:
        print("Abnormal - pred_score: {:.4f}".format(result.pred_score))


    image_bbox = result.image.copy()
    # Find the contours of the white mask
    contours, _ = cv2.findContours(result.pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create the bbox around the white contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_bbox, (x, y), (x+w, y+h), (255, 0, 0), 10)

    # stack three time the mask to simulate the three colour channels
    mask = cv2.merge((result.pred_mask,result.pred_mask,result.pred_mask))

    show_image_list(list_images=[result.image, result.heat_map, result.segmentations, mask, image_bbox],
                    list_titles=['image', 'heat_map', 'segmentations', 'mask', 'image_bbox'],
                    num_cols=3,
                    figsize=(20, 10),
                    grid=False,
                    title_fontsize=20,
                    path_image=path_result)