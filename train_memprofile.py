# train.py
import torch
import argparse
from tqdm import tqdm
from segment_anything import sam_model_registry
import numpy as np
from PIL import Image, ImageDraw
import wandb
import random
import os
import matplotlib.pyplot as plt
from torch.nn import functional as F

from data_loader import GuidedPromptDataset
from utils import CombinedLoss, dice_loss

# --- Visualization Utility Functions (Added) ---

def create_overlay_image(image_np, masks_list, points_list):
    """
    Creates an overlay image with all character masks colored randomly.
    (Operates on 1024x1024 input images/masks)
    
    Args:
        image_np: Original image as numpy array (1024x1024)
        masks_list: List of mask numpy arrays (1024x1024)
        points_list: List of prompt points (1024x1024 coordinates)
    """
    # Convert image to RGBA for alpha blending
    overlay_img = Image.fromarray(image_np).convert("RGBA")
    
    # Create an overlay layer
    overlay = Image.new('RGBA', overlay_img.size, (0, 0, 0, 0))
    
    for mask_np in masks_list:
        if mask_np.sum() == 0:  # Skip empty masks
            continue
            
        # Generate random color
        color = (
            random.randint(50, 255),
            random.randint(50, 255), 
            random.randint(50, 255),
            128  # 50% transparency
        )
        
        # Create colored mask
        mask_rgba = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
        mask_rgba[mask_np > 0] = color
        
        mask_pil = Image.fromarray(mask_rgba, 'RGBA')
        overlay = Image.alpha_composite(overlay, mask_pil)
    
    # Combine original image with overlay
    result = Image.alpha_composite(overlay_img, overlay)
    
    # Draw prompt points
    draw = ImageDraw.Draw(result)
    for point in points_list:
        point_x, point_y = point[0], point[1]
        radius = 3
        draw.ellipse(
            (point_x - radius, point_y - radius, point_x + radius, point_y + radius),
            fill='white',
            outline='black'
        )
    
    return result

def log_comparison_images(epoch, sam, train_loader, device):
    """
    Generate and log comparison images (GT vs. Prediction) for a single batch.
    """
    sam.eval() # Switch to evaluation mode for consistent logging
    
    # Take the first batch from the dataloader
    batch = next(iter(train_loader))
    
    images = batch['image'].to(device)
    gt_masks_256 = batch['gt_masks'] # List of [Num_Chars, 256, 256] tensors
    gt_points = batch['gt_points']   # List of [Num_Chars, 2] tensors
    
    logged_images = []

    with torch.no_grad():
        image_embeddings = sam.image_encoder(images)

        # Iterate through each image in the batch (typically just one)
        for i in range(len(images)):
            image_tensor = images[i]  # This should be [3, 1024, 1024]
            points_i = gt_points[i]
            masks_gt_256_i = gt_masks_256[i].to(device)
            
            # Debug: Print tensor shape to understand the issue
            print(f"image_tensor shape: {image_tensor.shape}")
            print(f"sam.pixel_mean shape: {sam.pixel_mean.shape}")
            print(f"sam.pixel_std shape: {sam.pixel_std.shape}")
            
            # Ensure image_tensor is 3D [C, H, W]
            if image_tensor.dim() > 3:
                # If tensor has extra dimensions, squeeze them
                image_tensor = image_tensor.squeeze()
            
            # Denormalize image tensor to numpy array for visualization
            mean = sam.pixel_mean.to(device)
            std = sam.pixel_std.to(device)
            
            # Ensure mean and std are broadcastable
            if mean.dim() == 1:
                mean = mean[:, None, None]
            if std.dim() == 1:
                std = std[:, None, None]
                
            image_np = (image_tensor * std + mean).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            # Upsample 256x256 GT masks to 1024x1024 for visualization
            gt_masks_1024_tensor = F.interpolate(
                masks_gt_256_i.unsqueeze(1), # Add channel dim
                size=(1024, 1024),
                mode='bilinear',
                align_corners=False
            ).squeeze(1) # Remove channel dim

            gt_masks_1024_list = [(m > 0.5).cpu().numpy() for m in gt_masks_1024_tensor]

            # Generate predictions for each prompt
            pred_masks_1024_list = []
            for j in range(len(points_i)):
                prompt_point = points_i[j].unsqueeze(0).unsqueeze(0).to(device)
                prompt_label = torch.tensor([[1]], device=device)
                target_mask_256 = masks_gt_256_i[j].unsqueeze(0).unsqueeze(0)

                sparse_emb, dense_emb = sam.prompt_encoder(
                    points=(prompt_point, prompt_label), boxes=None, masks=None
                )

                pred_masks_low_res, _ = sam.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=True,
                )

                # Select the best mask using the same logic as training
                dice_scores = torch.stack([1 - dice_loss(m, target_mask_256) for m in pred_masks_low_res.squeeze(0)])
                best_mask_idx = torch.argmax(dice_scores)
                
                # Postprocess the best low-res mask to full resolution
                predicted_mask_1024 = sam.postprocess_masks(
                    pred_masks_low_res[:, best_mask_idx, :, :].unsqueeze(1),
                    input_size=images[i].shape[-2:],
                    original_size=images[i].shape[-2:]
                )
                
                # Binarize and store
                predicted_mask_np = (predicted_mask_1024 > sam.mask_threshold).squeeze().cpu().numpy()
                pred_masks_1024_list.append(predicted_mask_np)

            # Create GT and Prediction overlay images
            points_list_np = points_i.cpu().numpy()
            
            random.seed(42 + i) # Use a consistent seed for matching colors
            gt_overlay = create_overlay_image(image_np, gt_masks_1024_list, points_list_np)
            
            random.seed(42 + i)
            pred_overlay = create_overlay_image(image_np, pred_masks_1024_list, points_list_np)

            # Concatenate images horizontally for comparison
            combined_img = Image.new('RGB', (gt_overlay.width * 2, gt_overlay.height))
            combined_img.paste(gt_overlay, (0, 0))
            combined_img.paste(pred_overlay, (gt_overlay.width, 0))
            
            # Save comparison image to local directory
            os.makedirs(args.comparison_images_dir, exist_ok=True)
            save_path = f"{args.comparison_images_dir}/epoch_{epoch}_image_{i+1}_comparison.png"
            combined_img.save(save_path)
            
            logged_images.append(wandb.Image(combined_img, caption=f"Epoch {epoch} - Image {i+1} (GT vs Pred)"))

    # Log the list of comparison images to wandb
    wandb.log({"comparison_images": logged_images, "epoch": epoch})
    
    sam.train() # Switch back to train mode

def custom_collate_fn(batch):
    """Custom collate function to handle variable number of characters per image."""
    images = []
    gt_masks = []
    gt_points = []
    
    for sample in batch:
        images.append(sample['image'])
        gt_masks.append(sample['gt_masks'])  # Keep as list - don't stack
        gt_points.append(sample['gt_points'])  # Keep as list - don't stack
    
    # Only stack images since they have fixed dimensions
    images = torch.stack(images, dim=0)
    
    return {
        'image': images,
        'gt_masks': gt_masks,  # List of tensors with different sizes
        'gt_points': gt_points  # List of tensors with different sizes
    }

def main(args):
    # Initialize wandb
    wandb.init(project="sam-finetuning", config=args)
    
    # Use the device argument with fallback to CPU
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Setup Model
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device)
    
    # Freeze image encoder
    for name, param in sam.named_parameters():
        if 'image_encoder' in name:
            param.requires_grad = False
        elif 'mask_decoder' in name:
            param.requires_grad = True
        elif 'prompt_encoder' in name:
            param.requires_grad = False
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

    # 2. Setup Optimizer and Loss
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, sam.parameters()),
        lr=args.lr
    )
    loss_fn = CombinedLoss()

    # 3. Setup Dataloader with custom collate function
    dataset = GuidedPromptDataset(
        index_file=args.index_file, 
        img_size=sam.image_encoder.img_size, 
        pixel_mean=sam.pixel_mean, 
        pixel_std=sam.pixel_std
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=custom_collate_fn,  # Add custom collate function
        num_workers=12
    )

    # 4. Training Loop
    for epoch in range(args.epochs):
        sam.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader):
            images = batch['image'].to(device) # [B, 3, 1024, 1024]
            gt_masks = batch['gt_masks'] # List of [Num_Chars, 256, 256]
            gt_points = batch['gt_points'] # List of [Num_Chars, 2]

            optimizer.zero_grad()

            with torch.no_grad():
                image_embeddings = sam.image_encoder(images)

            batch_loss = 0
            num_chars_in_batch = 0
            
            # Loop through each image in the batch
            for i in range(len(images)):
                points_i = gt_points[i]
                masks_i = gt_masks[i]
                
                # Loop through each character (prompt) in the image
                for j in range(len(points_i)):
                    prompt_point = points_i[j].unsqueeze(0).unsqueeze(0).to(device)
                    prompt_label = torch.tensor([[1]], device=device) # Foreground point
                    target_mask = masks_i[j].unsqueeze(0).unsqueeze(0).to(device) # [1, 1, 256, 256]
                    
                    num_chars_in_batch += 1

                    sparse_emb, dense_emb = sam.prompt_encoder(
                        points=(prompt_point, prompt_label), boxes=None, masks=None
                    )

                    pred_masks, pred_iou = sam.mask_decoder(
                        image_embeddings=image_embeddings[i].unsqueeze(0),
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=True,
                    ) # [1, 3, 256, 256]

                    # Find the best of the 3 predicted masks
                    dice_scores = torch.stack([1 - dice_loss(m, target_mask) for m in pred_masks.squeeze(0)])
                    best_mask_idx = torch.argmax(dice_scores)
                    best_pred_mask = pred_masks[:, best_mask_idx, :, :].unsqueeze(1)
                    
                    # Ensure target_mask is in [0,1] range for training
                    if target_mask.max() > 1.0:
                        target_mask = target_mask / 255.0
                        
                    best_save_pred_mask = sam.postprocess_masks(
                        best_pred_mask,
                        input_size=images[i].shape[-2:],
                        original_size=images[i].shape[-2:]
                    )
                    best_save_pred_mask = best_save_pred_mask > sam.mask_threshold
                    if best_save_pred_mask.max() > 1.0:
                        best_save_pred_mask = best_save_pred_mask.float() / 255.0
                    
                    loss = loss_fn(best_pred_mask, target_mask)
                    batch_loss += loss

            if num_chars_in_batch > 0:
                avg_loss = batch_loss / num_chars_in_batch
                avg_loss.backward()
                optimizer.step()
                epoch_loss += avg_loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_epoch_loss}")
        
        # Log metrics to wandb
        wandb.log({"epoch": epoch + 1, "loss": avg_epoch_loss})

        # --- MODIFICATION: Log comparison images at the end of the epoch ---
        log_comparison_images(epoch + 1, sam, train_loader, device)

        # Save model checkpoint
        checkpoint_path = f"{args.checkpoint_dir}/finetuned_sam_epoch_{epoch+1}.pth"
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(sam.state_dict(), checkpoint_path)
        
        # Log model checkpoint path to wandb
        wandb.log({"checkpoint_path": checkpoint_path})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='vit_b')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--index_file', type=str, default='master_index_250.json', help='Path to the JSON index file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1) # Start with 1 due to memory
    parser.add_argument('--comparison_images_dir', type=str, default='comparison_images_250', help='Directory to save comparison images')
    parser.add_argument('--checkpoint_dir', type=str, default='finetuned_models_250', help='Directory to save model checkpoints')
    args = parser.parse_args()
    main(args)