import torch
from torchcam.methods import SmoothGradCAMpp, LayerCAM
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import Dict, List
from torchcam.utils import overlay_mask
import numpy as np


# TODO: Maybe split the plots to other functions
def get_multiple_layers_result(model, img, input_tensor, layers: List[str], alpha):
    # Retrieve the CAM from several layers at the same time
    cam_extractor = LayerCAM(model, layers)

    # Preprocess your data and feed it to the model
    output = model(input_tensor.unsqueeze(0))
    # Retrieve the CAM by passing the class index and the model output
    cams = cam_extractor(output.squeeze(0).argmax().item(), output)

    cam_per_layer_list = []
    # Get the cam per target layer provided
    for cam in cams:
        cam_per_layer_list.append(cam.shape)

    print("The cams per target layer are: ", cam_per_layer_list)

    # Raw CAM
    _, axes = plt.subplots(1, len(cam_extractor.target_names))
    for id, name, cam in zip(
        range(len(cam_extractor.target_names)), cam_extractor.target_names, cams
    ):
        axes[id].imshow(cam.squeeze(0).numpy())
        axes[id].axis("off")
        axes[id].set_title(name)
    plt.show()

    fused_cam = cam_extractor.fuse_cams(cams)
    # Plot the raw version
    plt.imshow(fused_cam.squeeze(0).numpy())
    plt.axis("off")
    plt.title(" + ".join(cam_extractor.target_names))
    plt.show()
    # Plot the overlayed version
    result = overlay_mask(
        transforms.functional.to_pil_image(img),
        transforms.functional.to_pil_image(fused_cam, mode="F"),
        alpha=alpha,
    )
    plt.imshow(result)
    plt.axis("off")
    plt.title(" + ".join(cam_extractor.target_names))
    plt.show()

    cam_extractor.remove_hooks()


def get_localisation_mask(model, input_tensor, img):

    # Retrieve CAM for differnet layers at the same time
    cam_extractor = LayerCAM(model)
    output = model(input_tensor.unsqueeze(0))
    cams = cam_extractor(output.squeeze(0).argmax().item(), output)

    # Transformations
    resized_cams = [
        transforms.functional.resize(
            transforms.functional.to_pil_image(cam.squeeze(0)), img.shape[-2:]
        )
        for cam in cams
    ]
    segmaps = [
        transforms.functional.to_pil_image(
            (transforms.functional.resize(cam, img.shape[-2:]).squeeze(0) >= 0.5).to(
                dtype=torch.float32
            )
        )
        for cam in cams
    ]

    # Plots
    for name, cam, seg in zip(cam_extractor.target_names, resized_cams, segmaps):
        _, axes = plt.subplots(1, 2)
        axes[0].imshow(cam)
        axes[0].axis("off")
        axes[0].set_title(name)
        axes[1].imshow(seg)
        axes[1].axis("off")
        axes[1].set_title(name)
        plt.show()

    cam_extractor.remove_hooks()


def extract_cam(
    img,
    input_tensor,
    model,
    target_layer=None,
    localisation_mask: bool = True,
    multiple_layers: List[str] = [],
    alpha=0.5,
):
    cam_extractor = SmoothGradCAMpp(model, target_layer=target_layer)
    output = model(input_tensor.unsqueeze(0))
    # Get the CAM giving the class index and output
    cams = cam_extractor(output.squeeze(0).argmax().item(), output)

    cam_per_layer_list = []
    # Get the cam per target layer provided
    for cam in cams:
        cam_per_layer_list.append(cam.shape)

    print("The cams per target layer are: ", cam_per_layer_list)

    # The raw CAM
    for name, cam in zip(cam_extractor.target_names, cams):
        plt.imshow(cam.squeeze(0).numpy())
        plt.axis("off")
        plt.title(name)
        plt.show()

    # Overlayed on the image
    for name, cam in zip(cam_extractor.target_names, cams):
        result = overlay_mask(
            transforms.functional.to_pil_image(img),
            transforms.functional.to_pil_image(cam.squeeze(0), mode="F"),
            alpha=alpha,
        )
        plt.imshow(result)
        plt.axis("off")
        plt.title(name)
        plt.show()

    cam_extractor.remove_hooks()

    if localisation_mask:
        get_localisation_mask(model, input_tensor, img)

    if len(multiple_layers) > 0:
        get_multiple_layers_result(model, img, input_tensor, multiple_layers, alpha)