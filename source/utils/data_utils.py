import os
import random

import cv2

# import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# import seaborn as sns
import torch
import wandb

# from torch.utils.data import DataLoader
# from tqdm import tqdm


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)  # Set seed for Python's random module
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the device (GPU or CPU) that will be used for computation.

    Returns:
        torch.device: The device to be used for computation.
    """
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def initialize_wandb(args, use_wandb=True):
    """
    Initializes WandB for logging experiment metrics and hyperparameters.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        wandb.sdk.wandb_run.Run: WandB run object.

    """
    if use_wandb is False:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"
        print("Flag 'use_wandb' is set to False, WanDB will be disabled for this run!")
    else:
        print("Using WandB - Initializing run...")

    wandb.init(
        project=args.project_name,
        name=f"{args.model}-lr_{args.lr}-batch_{args.batch_size}-epochs_{args.epochs}-augs_{args.augs}-{args.timestamp}",
        group=args.model,
        tags=[
            "baseline",
            f"model:{args.model}",
        ],
        config=args,
    )

    return wandb


def tensor_to_numpy_img(img: torch.Tensor):
    """
    Convert a PyTorch tensor image to a NumPy array image.

    Args:
        img (torch.Tensor): The input tensor image.

    Returns:
        numpy.ndarray: The converted NumPy array image.
    """
    return img.permute(1, 2, 0).cpu().numpy()


def save_tensor_image(image, name):
    """
    Save a tensor image as a file.

    Args:
        image (torch.Tensor): The input tensor image.
        name (str): The name of the output file.

    Returns:
        None
    """
    image = image.permute(1, 2, 0)
    image = image.numpy()
    image = image - image.min()
    image = image / image.max()
    plt.imsave(name, image)


def save_image(image, title="Image", save_path="./", dpi=None, figsize=(10, 10)):
    """
    Save an image to a file with a title.

    Args:
        image (numpy.ndarray): The image to be saved.
        title (str, optional): The title of the image. Defaults to "Image".
        save_path (str, optional): The path to save the image file. Defaults to "./".
        dpi (int, optional): The resolution of the image in dots per inch. Defaults to None.
        figsize (tuple, optional): The figure size in inches. Defaults to (10, 10).
    """
    if dpi is not None:
        height, width, _ = image.shape
        figsize = (width / dpi, height / dpi)

    # Create figure with the specified or default size
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")

    # Save the image
    img_path = os.path.join(save_path, f"{title}.png")
    plt.savefig(img_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def show_image(image, title="Image", dpi=None, figsize=(10, 10)):
    """
    Display an image using matplotlib.

    Parameters:
    - image: numpy.ndarray
        The image to be displayed.
    - title: str, optional
        The title of the image plot. Default is "Image".
    - dpi: int, optional
        The resolution of the image in dots per inch. If not provided, the dpi will be calculated based on the image shape.
    - figsize: tuple, optional
        The size of the figure in inches. Default is (10, 10).

    Returns:
    None
    """
    if dpi is not None:
        height, width, _ = image.shape
        figsize = (width / dpi, height / dpi)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()
