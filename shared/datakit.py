from IPython import display
import json
import matplotlib
import matplotlib.pyplot as plt
import pathlib
from PIL import Image
import torch
from typing import IO
from torch import nn
from torchvision import models, transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import tempfile
import time
from upath import UPath


def download_data(
    dataset_class: VisionDataset,
    target_path: UPath,
    partition_configurations: dict[str, dict[str, str]],
    preserve_transparency: bool = False,
    override: bool = False
):
    """
        Downloads and extracts a Torchvision dataset into a structured directory format.

        This function downloads the raw data into a temporary directory, iterates 
        through the dataset, and renders the underlying PIL images into a standardized 
        folder structure: `<target_path>/<partition_folder>/<label>/image_<index>.<ext>`.
        It is designed to work universally across local and cloud filesystems using UPath.

        Args:
            dataset_class (Type[VisionDataset]): The Torchvision dataset class to instantiate 
                (e.g., torchvision.datasets.Flowers102). The instantiated dataset must yield 
                (PIL.Image, label) tuples.
            target_path (UPath): The universal path destination (local or cloud) where the 
                organized image directories will be created.
            partition_configurations (dict[str, dict[str, str]]): A mapping of desired output 
                folder names (e.g., "train", "val") to the keyword arguments required by the 
                `dataset_class` to initialize that specific split (e.g., {"split": "train"}).
            preserve_transparency (bool, optional): If True, saves images as PNGs to retain 
                alpha channels. If False, converts images to RGB and saves as JPEGs. 
                Defaults to False.
            override (bool, optional): If True, deletes and overwrites the `target_path` 
                if it already exists. If False, safely aborts the process to prevent 
                overwriting existing data. Defaults to False.

        Returns:
            None
    """
    if target_path.exists():
        if override:
            target_path.rmdir(recursive=True)
        else:
            print(f"Data already existed in {target_path}")
            return
    for partition_folder, dataset_configuration in partition_configurations.items():
        print(f"Downloading {partition_folder} from {dataset_class}")
        with tempfile.TemporaryDirectory() as holding_folder:
            dataset = dataset_class(
                root=holding_folder,
                download=True,
                **dataset_configuration
            )

            for index, (image, label) in enumerate(dataset):
                class_path = target_path / partition_folder / str(label)
                class_path.mkdir(parents=True, exist_ok=True)
                file_name = f"image_{index:05d}"
                if preserve_transparency:
                    format = "PNG"
                    file_name = f"{file_name}.png"
                else:
                    format = "JPEG"
                    image = image.convert("RGB")
                    file_name = f"{file_name}.jpg"
                destination_path = class_path / file_name
                with destination_path.open(mode="wb") as file:
                    image.save(file, format=format)
    print(f"Data load completed from {dataset_class} to {target_path}")


def data_to_image_folder(
    partition_configuration: list[tuple[str, str, str]],
    data_transforms: dict[str, transforms.transforms.Compose]
) -> dict[str, ImageFolder]:
    return {
        partition: ImageFolder(
            root=partition_folder,
            transform=data_transforms[partition_transform_key]
        )
        for partition, partition_folder, partition_transform_key in partition_configuration
    }


def dataset_to_dataloader(
    image_datasets: dict[str, ImageFolder],
    batch_size: int | None = None,
    shuffle: bool = True,
    num_workers: int = 4
) -> dict[str, DataLoader]:
    return{
        partition: DataLoader(
            dataset=partition_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        for partition, partition_dataset in image_datasets.items()
    }


def process_image(image: str | pathlib.Path | IO, input_size: int) -> torch.Tensor:
    """
    Loads and preprocesses an image for inference with a PyTorch model.

    This function reads an image file and applies a standard pipeline of 
    transformations to match the format the model expects. It resizes the 
    image, extracts a center crop, converts the pixel data into a PyTorch 
    tensor, and normalizes the color channels using standard ImageNet 
    mean and standard deviation values.

    Args:
        image (str, pathlib.Path, or file-like object): The file path or 
            binary stream of the image to be processed.
        input_size (int): The expected size of the input image

    Returns:
        torch.Tensor: The preprocessed image as a PyTorch tensor, typically 
            with shape (3, 224, 224) assuming an input_size of 224.
    """
    
    # Process a PIL image for use in a PyTorch model
    # We create a pipeline of transformations just like we did for the validation set
    transform = transforms.Compose(
        transforms=[
            # Resize the image so its shortest side is 256 pixels
            transforms.Resize(256),
            # Crop out the center 224x224 square, which is what the model expects
            transforms.CenterCrop(input_size),
            # Convert the image pixel values from integers (0-255) to floats (0-1)
            transforms.ToTensor(),
            # Normalize using the specific mean and standard deviation from ImageNet
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )
    # Open the image file, apply the transformations, and return the result
    return transform(Image.open(image))


def save_model_structures(output_file='data/model_structures.json'):
    """
    Iterates through all available torchvision models, retrieves their structure,
    and saves the string representation to a JSON file. This allows the analyzing of
    models to determine the last or classifier layer that will be replaced for transfer learning
    """
    # Dictionary to store the structure of each model
    model_structures = {}

    # Iterate through all available models
    for model_name in models.list_models():
        try:
            # Load the model
            # Note: weights="DEFAULT" downloads pre-trained weights which can be slow.
            # If you only need the architecture, you can use weights=None.
            model: nn.Module = models.get_model(
                name=model_name,
                weights="DEFAULT"
            )
            
            # Store the string representation of the model architecture
            model_structures[model_name] = str(model)
            
        except Exception as e:
            print(f"Skipping {model_name} due to error: {e}")

    # Write the accumulated structures to a JSON file
    with open(output_file, 'w') as f:
        json.dump(model_structures, f, indent=4)

    print(f"Model structures saved to {output_file}")


def update_plot(
    train_losses: list[float],
    valid_losses: list[float],
    epoch: int,
    ax: matplotlib.axes.Axes
):
    """
    Updates and displays a live plot of training and validation losses.

    This function is designed to be used inside a training loop (typically 
    within a Jupyter Notebook). It clears the provided matplotlib axes, 
    plots the updated loss curves, and dynamically refreshes the cell output 
    to create a real-time visualization of the model's training progress.

    Args:
        train_losses (list[float]): A list containing the history of training loss values.
        valid_losses (list[float]): A list containing the history of validation loss values.
        epoch (int): The current epoch index (0-indexed). The plot title will display `epoch + 1`.
        ax (matplotlib.axes.Axes): The matplotlib axes object where the plot will be drawn.

    Returns:
        None
    """
    ax.cla()
    ax.plot(train_losses, label="Training Loss", color="blue")
    ax.plot(valid_losses, label="Validation Loss", color="orange")
    ax.set_title(f"Training and Validation Loss (Epoch {epoch + 1})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")

    display.clear_output(wait=True)
    time.sleep(0.1)
    display.display(plt.gcf())

def load_json_to_dict(filepath: str) -> dict:
    with open(filepath, 'r') as file:
        return json.load(file)