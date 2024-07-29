import argparse
from datetime import datetime
import os
import torch
from utils import setup_logger, load_dataset
from unet3d import UNet3D
from train_unet3d import train_unet3d

def main(args):
    logger = setup_logger()
    args.logger = logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    logger.info(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading and preprocessing data...")
    dataset = load_dataset(args.data_dir, args.target_shape, args.patch_size, args.stride, args.new_spacing)
    
    # Split dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    model = UNet3D(in_channels=args.in_channels, out_channels=args.out_channels, features=args.features).to(device)

    train_unet3d(model, train_dataset, test_dataset, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D U-Net for MRI Segmentation")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for saving results")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--out_channels", type=int, default=1, help="Number of output channels")
    parser.add_argument("--features", type=int, default=32, help="Number of features in the first layer")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--target_shape", nargs=3, type=int, default=[128, 128, 128], help="Target shape for resampling")
    parser.add_argument("--new_spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0], help="New spacing for resampling")
    parser.add_argument("--patch_size", nargs=3, type=int, default=[32, 32, 32], help="Size of patches for training")
    parser.add_argument("--stride", nargs=3, type=int, default=[32, 32, 32], help="Stride for patch extraction")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for data loading")
    parser.add_argument("--save_interval", type=int, default=10, help="Epoch interval for saving checkpoints")
    parser.add_argument("--plot_interval", type=int, default=5, help="Epoch interval for plotting sample predictions")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    main(args)