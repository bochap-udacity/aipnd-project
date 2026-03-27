import argparse
import numpy as np

from shared.datakit import load_json_to_dict
from shared.modelkit import host_device, load_checkpoint, predict

def _get_input_args():
    parser = argparse.ArgumentParser(
        description="Image prediction application"
    )
    parser.add_argument(
        "image_path",
        help="Path of the image to be used for prediction"
    )
    parser.add_argument(
        "checkpoint_path",
        help="Path of the checkpoint file to load model parameters for prediction",
        default="checkpoints"
    )
    parser.add_argument(
        "--top_k",
        help="Number of top predictions to return",
        type=int,
        default=1
    )
    parser.add_argument(
        "--category_names_path",
        help="Path of the file to use for mapping category numbers to name",
        default="cat_to_name.json"
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
    image_path = args.image_path
    checkpoint_path = args.checkpoint_path
    category_names_path = args.category_names_path
    top_k = args.top_k
    is_gpu = args.is_gpu
    is_mps = args.is_mps
    input_size = 224
    device = host_device(
        is_gpu=is_gpu,
        is_mps=is_mps
    )    
    network, _ = load_checkpoint(
        filepath=checkpoint_path,
        device=device
    )
    model = network["model"]

    print("##### Prediction Parameters #####")
    print(f"image_path: {image_path}")
    print(f"checkpoint_path: {checkpoint_path}")
    print(f"category_names_path: {category_names_path}")
    print(f"top_k: {top_k}")
    print(f"is_gpu: {is_gpu}")
    print(f"is_mps: {is_mps}")
    print("-------------------------------")
    cat_to_name = load_json_to_dict(category_names_path)
    probs, labels = predict(image_path=image_path, model=model, input_size=input_size)
    probs = np.squeeze(probs)
    labels = np.squeeze(labels)
    class_names = [cat_to_name[str(int(label) + 1)] for label in labels]
    print(f"Top prediction is: {class_names[0]} with a probability of {probs[0]:.4f}")

if __name__ == "__main__":
    main()
