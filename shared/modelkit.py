import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import platform
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from typing import Any, Callable
from upath import UPath

from shared.datakit import process_image

PARTITION_CONFIGURATIONS={
    "train": { "split": "train" },
    "valid": { "split": "val" },
    "test": { "split": "test" },
}
_INPUT_SIZE = 224        # Pre-trained networks require images to be exactly 224x224 pixels

TRAINING_TRANSFORMERS = transforms.Compose([
    # Randomly crop a part of the image and resize it to 224x224 to make the model robust
    transforms.RandomResizedCrop(_INPUT_SIZE),
    # Randomly rotate the image up to 30 degrees so the model learns tilted flowers
    transforms.RandomRotation(30),
    # Randomly flip the image horizontally (mirror effect) for more variety
    transforms.RandomHorizontalFlip(),
    # Convert the image into a PyTorch Tensor (a mathematical grid of numbers the AI understands)
    transforms.ToTensor(),
    # Normalize the colors so the network can learn faster (using standard ImageNet values)
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))        
])

NON_TRAINING_TRANSFORMERS = transforms.Compose([
    # For non training, we don't want random rotations or flips. We just resize it.
    transforms.Resize(256),
    # Then cut out the exact center 224x224 square
    transforms.CenterCrop(_INPUT_SIZE),
    # Convert to Tensor
    transforms.ToTensor(),
    # Normalize the colors just like we did for training
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def host_device(
    is_gpu: bool | None = None,
    is_mps: bool | None = None
) -> torch.device:
    """
    Determines and returns the optimal hardware device for PyTorch operations.

    Checks for the availability of hardware accelerators in the following order:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon GPUs on macOS)
    3. CPU (Fallback if no compatible GPU is found)

    Returns:
        torch.device: The best available compute device ('cuda', 'mps', or 'cpu').
    """    
    if torch.cuda.is_available():
        if is_gpu or is_gpu is None:
            return torch.device("cuda")
    elif (platform.system() == "Darwin" and torch.mps.is_available()):
        if is_mps or is_mps is None:
            return torch.device("mps")
    return torch.device("cpu")


# Build and train your network
def build_network(
    model_name: str,
    out_features: int,
    hidden_features: int | None = None,
    freeze_layers: bool = True,
    optimizer_config: dict[str, Any] = {}
) -> dict[str, nn.Module]:
    """
    Builds a classifier using a pre-trained model from torchvision.models.

    Args:
        model_name (str): Name of the model to load (e.g., 'resnet18', 'vgg16').
            Supports only models with last layer named fc but can be expanded.
        out_features (int): Number of output features for the new classification layer.
        freeze_layers (bool, optional): Whether to freeze the pre-trained layers. Defaults to True.

    Returns:
        nn.Module: The modified model with the new classification layer.

    Raises:
        ValueError: If the model cannot be loaded or if the classification layer cannot be replaced automatically.
    """ 
    model: nn.Module = None
    optimizer: nn.Module = None
    criterion: nn.Module = None
    try:
        # Get the pre-trained architecture requested by the user
        model: nn.Module = models.get_model(
            name=model_name,
            weights="DEFAULT"
        )
    except ValueError as e:
        raise ValueError(f"Failed to load model '{model_name}': {e}")

    if freeze_layers:
        # Freezing the parameters means we will NOT update these weights during training.
        # We keep the smart feature-extraction abilities the model already learned.
        for param in model.parameters():
            param.requires_grad = False
    
    if hasattr(model, "fc"):
        # Model with fc as last layer
        # Example:
        #   (fc): Linear(in_features=512, out_features=1000, bias=True)
        # Extract the number of input features going into the old final layer
        in_features = model.fc.in_features
        # 1024 hidden_features works well for resnet as a default
        hidden_features = hidden_features if hidden_features is not None else 1024
        # Create our brand new classification layer (the part we will actually train)
        model.fc = nn.Sequential(
            # Connect the features to a hidden layer with hidden_features nodes
            nn.Linear(in_features=in_features, out_features=hidden_features, bias=True),
            # ReLU is an activation function that helps the network learn non-linear patterns
            nn.ReLU(),
            # Dropout randomly turns off 30% of neurons to prevent the model from memorizing the data (overfitting)
            nn.Dropout(p=0.3),
            # Final layer that outputs to our number of target classes (102 flowers)
            nn.Linear(in_features=hidden_features, out_features=out_features, bias=True)
        )
        # The optimizer is the tool that updates the weights based on how wrong the predictions were.
        # We only pass it the parameters of our NEW fully connected layer (model.fc).
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            **optimizer_config
        )
        # The criterion is the loss function used to calculate the error (CrossEntropy is standard for classification)
        criterion = nn.CrossEntropyLoss()
    elif hasattr(model, "classifier"):
        # Model with classifier as last layer
        # Example:
        #   (classifier): Sequential(
        #       (0): Linear(in_features=25088, out_features=4096, bias=True)
        #       (1): ReLU(inplace=True)
        #       (2): Dropout(p=0.5, inplace=False)
        #       (3): Linear(in_features=4096, out_features=4096, bias=True)
        #       (4): ReLU(inplace=True)
        #       (5): Dropout(p=0.5, inplace=False)
        #       (6): Linear(in_features=4096, out_features=1000, bias=True)
        #   )
        # Extract the number of input features going into the old final layer
        in_features = model.classifier[0].in_features
        # 4096 hidden_features works well for vgg as a default
        hidden_features = hidden_features if hidden_features is not None else 4096        
        # Create our brand new classification layer (the part we will actually train)
        model.classifier = nn.Sequential(
            # Connect the features to a hidden layer with hidden_features nodes top compress information
            # from large features 25088 for vgg19 to 102
            nn.Linear(in_features=in_features, out_features=hidden_features),
            # ReLU is an activation function that helps the network learn non-linear patterns
            nn.ReLU(),
            # Dropout randomly turns off 40% of neurons to prevent the model from memorizing the data (overfitting)
            nn.Dropout(p=0.4),
            # Final layer that outputs to our number of target classes (102 flowers)
            nn.Linear(in_features=hidden_features, out_features=out_features),
        )
        # The optimizer is the tool that updates the weights based on how wrong the predictions were.
        # We only pass it the parameters of our NEW fully connected layer (model.fc).
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            weight_decay=0.0001,
            **optimizer_config
        )
        # The criterion is the loss function used to calculate the error (CrossEntropy is standard for classification)
        criterion = nn.CrossEntropyLoss()
    else:
        raise(
            f"Cannot automatically replace the classification layer for {model_name}. "
            "The model structure is unsupported by this script."            
        )    


    return {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "hidden_features": hidden_features,
        "out_features": out_features
    }


# Save the checkpoint
def save_checkpoint(
    model_name: str,
    model: nn.Module,
    train_data: Dataset,
    min_valid_loss: float,
    filename: str = 'checkpoint.pth',
    out_features: int = 102,
    hidden_features: int = 512
):
    """
    Saves a trained PyTorch model's state and metadata to a checkpoint file.
    
    This function creates a comprehensive dictionary containing the model's 
    learned parameters (state_dict) alongside essential architectural details 
    and the class-to-index mapping. Saving these details ensures the model 
    can be accurately rebuilt and used for inference later without needing 
    the original training code or dataset.

    Args:
        model_name (str): The name of the base architecture (e.g., 'resnet50', 'vgg16').
        model (nn.Module): The trained PyTorch neural network model.
        train_data (Dataset): The training dataset object, used to extract the 
            `class_to_idx` dictionary mapping class names to integer labels.
        min_valid_loss (float): The best validation loss achieved during training, 
            useful for tracking model performance over time.
        filename (str, optional): The file path and name where the checkpoint 
            will be saved. Defaults to 'checkpoint.pth'.
        out_features (int, optional): The number of output features for the model's 
            final classifier layer. Defaults to 102.
        hidden_features (int, optional): The number of hidden features for the model's 
            final classifier layer. Defaults to 512.
    Returns:
        None
    """
    # Save the mapping of classes to indices
    model.class_to_idx = train_data.class_to_idx
    
    checkpoint = {
        'model_name': model_name,
        'hidden_features': hidden_features,
        'out_features': out_features,
        'min_valid_loss': min_valid_loss,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict()
    }
    file_path = UPath(filename)
    # ensure that the parent folders exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(
    filepath: str,
    device: torch.device
) -> tuple[dict[str, Any], float]:
    """
    Loads a saved PyTorch checkpoint and reconstructs the model.
    
    This function reads a previously saved checkpoint file, extracts the 
    architectural metadata (like the model name and output features), and uses 
    it to rebuild the network via the `build_network` function. It then 
    restores the learned weights (state_dict) and the class-to-index mapping, 
    ensuring the model is ready for inference or further training.

    Args:
        filepath (str): The file path to the saved checkpoint (e.g., 'checkpoint.pth').
        device (torch.device): The computation device ('cpu', 'cuda', or 'mps') 
            to which the rebuilt model will be moved.
            
    Returns:
        tuple[dict[str, Any], float]: A tuple containing:
            - A dictionary containing the rebuilt network architecture 
              (typically keyed with "model" and other components).
            - The minimum validation loss recorded when the checkpoint was saved.
    """
    checkpoint = torch.load(filepath, map_location=device)
    # Rebuild the model architecture
    network = build_network(
        model_name=checkpoint['model_name'],
        out_features=checkpoint['out_features'],
        hidden_features=checkpoint['hidden_features'],
        freeze_layers=True
    )
    network["model"].to(device)
    # Load the weights
    network["model"].load_state_dict(checkpoint['state_dict'])
    # Load the class mapping
    network["model"].class_to_idx = checkpoint['class_to_idx']
    
    return network, checkpoint["min_valid_loss"]

def train_model(
    model: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    optimizer: nn.Module,
    device: torch.device,
    num_epochs: int,
    min_valid_loss: float = np.inf,
    plotter: Callable[[list[float], list[float], int, matplotlib.axes.Axes], None] = None
) -> tuple[nn.Module, float]:
    """
    Trains a PyTorch model, performing validation and live plotting per epoch.

    This function executes the standard PyTorch training loop. For each epoch, 
    it performs a training phase (forward pass, loss calculation, backward 
    pass, optimization) followed by a validation phase (evaluation without 
    gradients). It calculates and prints the loss and accuracy metrics, 
    updates a live visualization, and tracks the best validation loss.

    Args:
        model (nn.Module): The PyTorch neural network model to be trained.
        training_loader (DataLoader): The DataLoader providing training data batches.
        validation_loader (DataLoader): The DataLoader providing validation data batches.
        criterion (nn.Module): The loss function used to compute the error (e.g., nn.CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): The optimization algorithm used to update model weights.
        device (torch.device): The computation device ('cpu', 'cuda', or 'mps').
        num_epochs (int): The total number of training epochs.
        min_valid_loss (float, optional): The initial minimum validation loss for tracking 
            improvements. Defaults to infinity.
        plotter (Callable[[list[float], list[float], int, matplotlib.axes.Axes], None], Optional): Callable that plots the losses of the training to a graph. Defailts to None

    Returns:
        tuple[nn.Module, float]: A tuple containing:
            - The trained model (with a newly attached `class_to_idx` attribute).
            - The minimum validation loss achieved during training.
    """
    model.to(device)
    train_losses, valid_losses = [], []
    _, ax = plt.subplots(figsize=(8, 5))
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        # Reset metrics for new epoch
        train_loss = 0.0
        train_accuracy = 0.0
        valid_loss = 0.0
        valid_accuracy = 0.0
        
        # --- TRAINING PHASE ---
        # Switch the model to training mode (enables dropout and batch norm)
        model.train()
        for inputs, labels in training_loader:
            # Move our images and labels to the designated device (GPU or CPU)
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Clear the old gradients from the last step (otherwise they accumulate)
            optimizer.zero_grad()
            
            # Forward pass: the model guesses what the images are
            logps = model(inputs)
            # Calculate the loss (the error between the guesses and the real labels)
            batch_train_loss = criterion(logps, labels)

            # Calculate accuracy of our model on the validation set
            # topk(1) gets the highest probability class for each image
            _, top_class = logps.topk(1, dim=1)
            # Check if the predicted class matches the true label
            equals = top_class == labels.view(*top_class.shape)
            # Convert booleans to floats and take the mean to get the accuracy percentage
            train_accuracy += torch.sum(equals.type(torch.FloatTensor))
            # Keep a running total of the training loss for this epoch
            train_loss += batch_train_loss.item() * inputs.size(0)
            # Backward pass: compute the gradients (how much each weight contributed to the error)
            batch_train_loss.backward()
            # Optimizer step: update the weights to make better guesses next time
            optimizer.step()

            
        # --- VALIDATION PHASE ---
        # Switch model to evaluation mode (turns off dropout so predictions are consistent)
        model.eval()
        # Turn off gradients since we are not training (saves memory and speeds up computation)
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model(inputs)
                batch_valid_loss = criterion(logps, labels)

                # Calculate accuracy of our model on the validation set
                # topk(1) gets the highest probability class for each image
                _, top_class = logps.topk(1, dim=1)
                # Check if the predicted class matches the true label
                equals = top_class == labels.view(*top_class.shape)
                # Convert booleans to floats and take the mean to get the accuracy percentage
                valid_accuracy += torch.sum(equals.type(torch.FloatTensor))
                valid_loss += batch_valid_loss.item()  * inputs.size(0)
        
        # Calculate average losses
        epoch_train_loss = train_loss / len(training_loader.dataset)
        epoch_train_accuracy = (train_accuracy / len(training_loader.dataset)) * 100
        epoch_valid_loss = valid_loss / len(validation_loader.dataset)
        epoch_valid_accuracy = (valid_accuracy / len(validation_loader.dataset)) * 100
        
        epoch_duration_s = time.time() - epoch_start_time

        train_losses.append(epoch_train_loss)
        valid_losses.append(epoch_valid_loss)
        
        # Update the plot
        if plotter:
            plotter(
                train_losses=train_losses,
                valid_losses=valid_losses,
                epoch=epoch,
                ax=ax
            )

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Time: {epoch_duration_s:.2f}s, "
              f"Train loss: {epoch_train_loss:.3f}, "
              f"Train accuracy: {epoch_train_accuracy:.3f}%, "
              f"Validation loss: {epoch_valid_loss:.3f}, "
              f"Validation accuracy: {epoch_valid_accuracy:.3f}%")
                
        if epoch_valid_loss < min_valid_loss:
            min_valid_loss = epoch_valid_loss
            print("Validation loss has decreased.")

    model.class_to_idx = training_loader.dataset.class_to_idx
    return (model, min_valid_loss)

def predict(
    image_path: str,
    model: nn.Module,
    input_size: int,
    topk=5,
    device: torch.device = torch.device("cpu")
):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    
    Args:
        image_path (str): The file path to the image we want to classify.
        model (nn.Module): Our fully trained neural network.
        input_size (int): The expected size of the input image.
        topk (int): How many of the most likely predictions to return (default is 5).
        
    Returns:
        top_prob (list): The probabilities of the top K classes.
        top_classes (list): The actual names of the top K classes.
    """
    
    # Implement the code to predict the class from an image file
    model.to(device=device)
    # Put model into evaluation mode (turns off dropout)
    model.eval()

    # Load and process the image into the required format
    image = process_image(image=image_path, input_size=input_size)
    image.to(device=device)
    # PyTorch models expect a batch dimension. unsqueeze(0) changes the image shape 
    # from [channels, height, width] to [1, channels, height, width] (a batch of size 1)
    image = image.unsqueeze(0)

    # Pass the image through the model
    output = model.forward(image)
    # Get the top K probabilities and their corresponding indices
    top_prob, top_class = torch.topk(input=output, k=topk)
    # Since our model outputs log-probabilities (LogSoftmax), we use .exp() to get the actual probabilities between 0 and 1
    top_prob = top_prob.exp()
    # Detach from the graph and convert to numpy arrays for easier handling
    top_prob = top_prob.unsqueeze(0).detach().numpy()
    top_class = top_class.detach().squeeze().numpy()

    # Invert the class_to_idx dictionary so we can look up the class label from the index
    idx_to_class = {
        val: key for key, val in model.class_to_idx.items()
    }
    # Map the predicted indices to their string class labels
    top_classes = [
        idx_to_class[i] for i in top_class
    ]
    return top_prob, top_classes