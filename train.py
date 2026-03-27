import argparse
import numpy as np
from torchvision.datasets import Flowers102
from upath import UPath


from shared.datakit import data_to_image_folder, dataset_to_dataloader, download_data
from shared.modelkit import NON_TRAINING_TRANSFORMERS, PARTITION_CONFIGURATIONS, TRAINING_TRANSFORMERS, build_network, host_device, save_checkpoint, train_model

def _get_input_args():
    parser = argparse.ArgumentParser(
        description="Image classification training application"
    )
    parser.add_argument(
        "data_dir",
        help="Directory containing the data used for transfer learning training"
    )
    parser.add_argument(
        "--save_dir",
        help="Directory to save checkpoint files in training",
        default="checkpoints"
    )
    parser.add_argument(
        "--arch",
        help="Souce model used for transfer learning",
        default="resnet152"
    )
    parser.add_argument(
        "--learning_rate",
        help="Learning rate used during model transfer learning training",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--hidden_units",
        help="Number of units used for hidden layer during training",
        type=int,
        default=None
    )
    parser.add_argument(
        "--epochs",
        help="Number of epochs used for training",
        type=int,
        default=5
    )
    parser.add_argument('--gpu', action='store_true',
        default=False,
        dest='is_gpu',
        help='Set the GPU mode to true'
    )
    parser.add_argument('--mps', action='store_true',
        default=False,
        dest='is_mps',
        help='Set the MPS mode to true'
    )
    return parser.parse_args()

def main():
    args = _get_input_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    is_gpu = args.is_gpu
    is_mps = args.is_mps
    batch_size = 32
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    device = host_device(
        is_gpu=is_gpu,
        is_mps=is_mps
    )

    print("##### Training Parameters #####")
    print(f"data_dir: {data_dir}")
    print(f"save_dir: {save_dir}")
    print(f"arch: {arch}")
    print(f"batch_size: {batch_size}")
    print(f"learning_rate: {learning_rate}")
    print(f"hidden_units: {hidden_units}")
    print(f"epochs: {epochs}")
    print(f"is_gpu: {is_gpu}")
    print(f"is_mps: {is_mps}")
    print("-------------------------------")
    data_transforms = {
        "train": TRAINING_TRANSFORMERS,
        "val": NON_TRAINING_TRANSFORMERS,
        "test": NON_TRAINING_TRANSFORMERS
    }

    download_data(
        partition_configurations=PARTITION_CONFIGURATIONS,
        dataset_class=Flowers102,
        target_path=UPath(data_dir),
    )

    # Load the datasets with ImageFolder. This looks at folders and automatically assumes 
    # the folder name is the label (class) for the images inside it.
    image_datasets = data_to_image_folder(
        partition_configuration = [
            ("training", train_dir, "train"),
            ("validation", valid_dir, "val"),
            ("testing", test_dir, "test"),
        ],
        data_transforms=data_transforms
    )

    # DataLoaders wrap the datasets and act as iterators. They hand out images 
    # in batches (e.g., 32 at a time) and shuffle them so the model doesn't memorize the order.
    dataloaders = dataset_to_dataloader(
        image_datasets=image_datasets,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )    
    # Build and train your network
    network = build_network(
        model_name=arch,
        hidden_features=hidden_units, 
        out_features=102,
        optimizer_config={
            "lr": learning_rate
        }
    )

    _, min_valid_loss = train_model(
        model=network["model"],
        criterion=network["criterion"],
        optimizer=network["optimizer"],
        training_loader=dataloaders["training"],
        validation_loader=dataloaders["validation"],
        device=device,
        num_epochs=epochs,
        min_valid_loss=np.inf,
    )
    save_checkpoint(
        model_name=arch,
        model=network["model"],
        train_data=image_datasets["training"],
        min_valid_loss=min_valid_loss,
        filename=f"{save_dir}/{arch}_{epochs}_{batch_size}_{learning_rate}.pth",
        hidden_features=network["hidden_features"],
        out_features=network["out_features"],
    )

if __name__ == "__main__":
    main()
