# Project Documentation: Image Classifier with Deep Learning

This project consists of a development environment (Jupyter Notebook) and a functional command-line application for training and using a deep neural network to classify images of flowers. The system utilizes transfer learning with PyTorch to achieve high accuracy on a dataset of 102 flower categories.

## Folder Structure

```text
.
├── Image Classifier Project.ipynb   # Development and prototyping notebook
├── train.py                         # CLI script to train a new network
├── predict.py                       # CLI script to predict classes from an image
├── cat_to_name.json                 # Mapping from category label to flower name
├── shared/                          # Modular logic for data and model management
│   ├── datakit.py                   # Data loading and augmentation utilities
│   └── modelkit.py                  # Model building, training, and inference logic
└── assets/                          # Supporting files and documentation images
```

## Shared Logic Implementation

To ensure consistency between Part 1 (Notebook) and Part 2 (CLI), core functionality is modularized in the `shared` directory.

### `shared/datakit.py`
This module handles all data-related operations:
* **Augmentation:** Uses `torchvision.transforms` to perform random scaling, rotations, mirroring, and cropping on training data.
* **Normalization:** Standardizes images to the mean and standard deviation required by pre-trained Torchvision models.
* **Loading:** Implements `ImageFolder` for directory-based labeling and `DataLoader` for batching (default batch size: 64).

### `shared/modelkit.py`
This module manages the neural network lifecycle:
* **Architecture Loading:** Supports loading pre-trained models (e.g., VGG16, ResNet).
* **Classifier Definition:** Defines a custom feedforward network that serves as the new "head" for the pre-trained features.
* **Checkpointing:** Functions to save and load `.pth` files including weights, architecture name, class-to-index mapping, and hyperparameters.
* **Inference:** Contains `process_image` (converting PIL images to Tensors) and `predict` (top-K probability calculation).

---

## Part 1: Development Notebook

The notebook follows the implementation criteria for an end-to-end machine learning pipeline:

1.  **Package Imports:** All dependencies (PyTorch, Torchvision, Matplotlib, NumPy) are imported in the initial cell.
2.  **Data Processing:** * Training data is augmented with random transforms.
    * All data sets (train, validation, test) are normalized and cropped to 224x224 pixels.
3.  **Transfer Learning:**
    * A pre-trained VGG16 model is loaded.
    * Feature parameters are frozen (`param.requires_grad = False`).
    * A custom classifier with ReLU activation and Dropout is attached.
4.  **Training & Validation:**
    * The network is trained using the features as input.
    * Validation loss and accuracy are printed during the training loop.
5.  **Evaluation:**
    * Final accuracy is measured on the unseen test dataset.
6.  **Sanity Check:**
    * A visualization function uses `matplotlib` to display a test image and a bar chart of the top 5 predicted flower names.

---

## Part 2: Command Line Application

The application is split into two specialized scripts utilizing `argparse` for flexibility.

### 1. Training (`train.py`)
Successfully trains a new network on a directory of images and saves the result.

* **Usage:** `python train.py data_directory`
* **Validation:** Prints training loss, validation loss, and validation accuracy in real-time.
* **Key Options:**
    * `--save_dir`: Directory to store the checkpoint.
    * `--arch`: Choose between architectures (e.g., `vgg13`, `resnet18`).
    * `--learning_rate`, `--hidden_units`, `--epochs`: Tune hyperparameters.
    * `--gpu`: Enables training on CUDA or MPS devices.

### 2. Inference (`predict.py`)
Predicts the flower name and probability from a single image.

* **Usage:** `python predict.py /path/to/image checkpoint`
* **Mapping:** Displays human-readable flower names if a `--category_names` JSON file is provided.
* **Key Options:**
    * `--top_k`: Returns the top $K$ most likely classes.
    * `--gpu`: Executes inference on a GPU for faster results.
