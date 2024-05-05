import gradio as gr
import argparse
from anomalib.deploy import TorchInferencer
import cv2
import numpy as np
from torch import as_tensor
from torchvision.transforms.v2.functional import to_dtype, to_image
import torch

font = cv2.FONT_ITALIC


def predict(image):
    result = inferencer.predict(image=image)

    #print("inferencer.metadata", inferencer.metadata)
    print("pred_score: {:.4f} - pred_label: {}".format(result.pred_score, result.pred_label))
    original_image = result.image.copy()

    # result.pred_score gives the score to be anomalous
    if result.pred_label == 0:
        normal_score = 1 - result.pred_score
        print("Normal - pred_score: {:.4f}".format(normal_score))
        color = (0, 255, 0)
        text = "Normal - score: " + str(round(normal_score, 4))
    else:
        print("Abnormal - pred_score: {:.4f}".format(result.pred_score))
        color = (255, 0, 0)
        text = "Abormal - score: " + str(round(result.pred_score, 2))
    cv2.putText(original_image, text, (0, 250), font, 0.7, color, 2)

    image_bbox = result.image.copy()
    # Find the contours of the white mask
    contours, _ = cv2.findContours(result.pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create the bbox around the white contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_bbox, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # stack three time the mask to simulate the three colour channels
    mask = cv2.merge((result.pred_mask,result.pred_mask,result.pred_mask))

    print("----------------------------------------------------------------------")

    return original_image, result.heat_map, result.segmentations, mask, image_bbox


def show_image(image):

    print("type(image): ", type(image))

    image = image.convert("RGB")
    image = image.resize((256, 256))
    image = to_dtype(to_image(image), torch.float32, scale=True) if as_tensor else np.array(image) / 255.0

    image, heat_map, segm, mask, bbox = predict(image)

    return image, heat_map, segm, mask, bbox


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_torch_model', type=str, help='Path to the torch model')
    #parser.add_argument("--port", type=int, default=None)
    #parser.add_argument("--server", type=str, default='0.0.0.0')

    args = parser.parse_args()
    path_torch_model = args.path_torch_model

    # load model
    print("load model")
    inferencer = TorchInferencer(path=path_torch_model, device="auto")

    app = gr.Interface(
        fn=show_image,
        inputs=gr.Image(label="Input Image Component", type="pil", height=512, width=512),
        outputs=[gr.Image(label="Original Image", height=512, width=512),
                 gr.Image(label="Heat Map", height=512, width=512),
                 gr.Image(label="Segmentation", height=512, width=512),
                 gr.Image(label="Mask", height=512, width=512),
                 gr.Image(label="Bounding box", height=512, width=512)]
    )

    app.launch()