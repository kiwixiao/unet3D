import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from unet3d import UNet3D
from utils import plot_prediction, calculate_metrics, load_dataset
from torch.utils.tensorboard import SummaryWriter
import random

def train_unet3d(model, train_dataset, test_dataset, args):
    device = args.device
    logger = args.logger
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard_logs'))

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    epochs = []
    train_losses = []
    val_dice_scores = []
    val_iou_scores = []

    for epoch in range(args.num_epochs):
        model.train()
        
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for i, (mri_patch, mask_patch) in enumerate(pbar):
            mri_patch, mask_patch = mri_patch.to(device), mask_patch.to(device)

            optimizer.zero_grad()
            
            outputs = model(mri_patch)
            loss = criterion(outputs, mask_patch)
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_dice = 0
        val_iou = 0
        num_val_samples = 0

        with torch.no_grad():
            for mri_patch, mask_patch in test_loader:
                mri_patch, mask_patch = mri_patch.to(device), mask_patch.to(device)
                pred_mask = model(mri_patch)
                pred_mask = (pred_mask > 0).float()
                metrics = calculate_metrics(mask_patch.cpu().numpy(), pred_mask.cpu().numpy())
                val_dice += metrics['F1']
                val_iou += metrics['IoU']
                num_val_samples += 1

        avg_val_dice = val_dice / num_val_samples
        avg_val_iou = val_iou / num_val_samples

        logger.info(f"Validation - Dice: {avg_val_dice:.4f}, IoU: {avg_val_iou:.4f}")

        # Log to TensorBoard
        writer.add_scalar('Loss/Train', avg_loss, epoch)
        writer.add_scalar('Metrics/Dice', avg_val_dice, epoch)
        writer.add_scalar('Metrics/IoU', avg_val_iou, epoch)

        # Save model checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))

            # Plot and save random sample predictions
            random_idx = random.randint(0, len(test_dataset) - 1)
            sample_mri, sample_mask = test_dataset[random_idx]
            sample_mri = sample_mri.unsqueeze(0).to(device)
            sample_mask = sample_mask.unsqueeze(0).to(device)
            with torch.no_grad():
                sample_pred = model(sample_mri)
            plot_prediction(sample_mri.squeeze().cpu().numpy(), 
                            sample_mask.squeeze().cpu().numpy(), 
                            sample_pred.squeeze().cpu().numpy(), 
                            epoch, args.output_dir)

        # Append metrics for final plotting
        epochs.append(epoch + 1)
        train_losses.append(avg_loss)
        val_dice_scores.append(avg_val_dice)
        val_iou_scores.append(avg_val_iou)

    writer.close()

    # Plot and save training curves
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'training_loss.png'))
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, val_dice_scores, label='Dice Score')
    plt.plot(epochs, val_iou_scores, label='IoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'validation_metrics.png'))
    plt.close()

    logger.info("Training completed.")