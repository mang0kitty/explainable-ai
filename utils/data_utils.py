from typing import Dict, Any
import yaml
from torchvision import transforms
from torchvision import datasets
from torch.utils import data
from torchvision.io.image import read_image


def load_conf(
    model_trans_params_name: str = "ImageNet_transformation",
    path: str = "utils/conf.yaml",
) -> Dict[str, Dict[str, Any]]:
    """
    Loads params base on yaml's path and spesific model's
    transformation params

    Args:
        model_trans_params (str): The name of the params for the desired input
        transformation of the model. Defaults to "ImageNet_transformation"

        path (str): The path where the yaml file is located. Defaults to "utils/conf.yaml".

    Returns:
        dict: params dictionary

    """

    with open(path, "r") as stream:
        params = yaml.safe_load(stream)

        if model_trans_params_name in params.keys():
            return params[model_trans_params_name]
        else:
            return {}


def root_load_and_transform_image(
    trans_params: Dict[str, Dict[str, Any]], root: str = "examples/00_quick_start/data/"
):
    """Loads an image from a root path and transforms it based on specific params.
    ImageFolder needs a specific folders structure to be used. This function can be customised
    for reading more images as batches.

    Args:
        trans_params (dict): the specific transformation params
        root (str): The path that leads to the structured path of datasets.
        Defaults to "examples/00_quick_start/data/".

    Returns:
        input_tensor: The tensor to be input in the model algorithm
    """

    resize = trans_params["Resize"]
    normalize = trans_params["Normalize"]

    transform = transforms.Compose(
        [
            transforms.Resize((resize["h"], resize["w"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize["mean"], std=normalize["std"]),
        ]
    )

    # define only one image dataset and load it
    dataset = datasets.ImageFolder(root=root, transform=transform)
    data_loader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)
    input_tensor, _ = next(iter(data_loader))

    return input_tensor


def transform_image(
    img,
    trans_params,
):
    """Loads an image from a any path and transforms it based on specific params.

    Args:
        trans_params (dict): the specific transformation params

    Returns:
        input_tensor: The tensor to be input in the model algorithm
    """

    resize = trans_params["Resize"]

    normalize = trans_params["Normalize"]
    input_tensor = transforms.functional.normalize(
        transforms.functional.resize(img, (resize["h"], resize["w"])) / 255.0,
        normalize["mean"],
        normalize["std"],
    )

    return input_tensor