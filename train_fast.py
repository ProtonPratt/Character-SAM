# train_fast.py
import torch
import argparse
from tqdm import tqdm
from segment_anything import sam_model_registry
import numpy as np
from PIL import Image, ImageDraw
import random
import wandb
import os

from data_loader import GuidedPromptDataset
from utils import CombinedLoss, dice_loss

def batched_dice_loss(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    
    # Flatten label and prediction tensors
    preds = preds.view(preds.shape[0], preds.shape[1], -1)
    targets = targets.view(targets.shape[0], targets.shape[1], -1)
    
    intersection = (preds * targets).sum(-1)
    dice = (2. * intersection + smooth) / (preds.sum(-1) + targets.sum(-1) + smooth)
    
    return 1 - dice

def create_overlay_image(original_np, masks_list, points_list, output_path, title=""):
    """
    Creates an overlay image with all character masks colored randomly.
    
    Args:
        original_np: Original image as numpy array
        masks_list: List of mask numpy arrays
        points_list: List of prompt points
        output_path: Where to save the result
        title: Title prefix for filename
    """
    # Create a copy of the original image
    overlay_img = Image.fromarray(original_np).convert("RGBA")
    
    # Create an overlay layer
    overlay = Image.new('RGBA', overlay_img.size, (0, 0, 0, 0))
    
    for i, (mask_np, point) in enumerate(zip(masks_list, points_list)):
        if mask_np.sum() == 0:  # Skip empty masks
            continue
            
        # Generate random color
        color = (
            random.randint(50, 255),
            random.randint(50, 255), 
            random.randint(50, 255),
            128  # 50% transparency
        )
        
        # Resize mask to match overlay image size if needed
        if mask_np.shape[:2] != (overlay_img.size[1], overlay_img.size[0]):
            mask_pil_temp = Image.fromarray((mask_np * 255).astype(np.uint8), 'L')
            mask_pil_temp = mask_pil_temp.resize(overlay_img.size, Image.NEAREST)
            mask_np = np.array(mask_pil_temp) > 0
        
        # Convert mask to RGBA with the random color
        mask_colored = np.zeros((overlay_img.size[1], overlay_img.size[0], 4), dtype=np.uint8)
        mask_colored[mask_np > 0] = color
        
        mask_pil = Image.fromarray(mask_colored, 'RGBA')
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
    
    # Convert back to RGB for saving
    result_rgb = Image.new('RGB', result.size, (255, 255, 255))
    result_rgb.paste(result, mask=result.split()[-1])
    
    result_rgb.save(output_path)
    return result_rgb

def create_comparison_image(original_img, gt_overlay, pred_overlay, output_path):
    """
    Combines original image, ground truth overlay, and prediction overlay into a single image.
    """
    width, height = original_img.size
    comparison_img = Image.new('RGB', (width * 3, height))
    comparison_img.paste(original_img, (0, 0))
    comparison_img.paste(gt_overlay, (width, 0))
    comparison_img.paste(pred_overlay, (width * 2, 0))
    
    comparison_img.save(output_path)

def custom_collate_fn(batch):
    """Custom collate function to handle variable number of characters per image."""
    images = []
    gt_masks = []
    gt_points = []
    image_paths = []
    gt_full_masks = []
    
    for sample in batch:
        images.append(sample['image'])
        gt_masks.append(sample['gt_masks'])  # Keep as list - don't stack
        gt_points.append(sample['gt_points'])  # Keep as list - don't stack
        image_paths.append(sample['image_path'])
        if 'gt_full_masks' in sample:
            gt_full_masks.append(sample['gt_full_masks'])
    
    # Only stack images since they have fixed dimensions
    images = torch.stack(images, dim=0)
    
    output = {
        'image': images,
        'gt_masks': gt_masks,  # List of tensors with different sizes
        'gt_points': gt_points,  # List of tensors with different sizes
        'image_path': image_paths
    }

    if gt_full_masks:
        output['gt_full_masks'] = gt_full_masks

    return output

def main(args):
    # Initialize wandb
    wandb.init(project="sam-finetuning-fast", config=args)
    
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
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
        index_file='master_index_250.json', 
        img_size=sam.image_encoder.img_size, 
        pixel_mean=sam.pixel_mean, 
        pixel_std=sam.pixel_std,
        return_path=True,
        return_full_mask=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=custom_collate_fn,  # Add custom collate function
        num_workers=12
    )

    # 4. Training Loop
    embedding_cache = {}
    for epoch in range(args.epochs):
        sam.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader):
            images = batch['image'].to(device) # [B, 3, 1024, 1024]
            gt_masks = batch['gt_masks'] # List of [Num_Chars, 256, 256]
            gt_points = batch['gt_points'] # List of [Num_Chars, 2]
            image_paths = batch['image_path']

            optimizer.zero_grad()

            # Get image embeddings from cache or compute them
            image_embeddings = []
            for i, image_path in enumerate(image_paths):
                if image_path in embedding_cache:
                    image_embeddings.append(embedding_cache[image_path].to(device))
                else:
                    with torch.no_grad():
                        embedding = sam.image_encoder(images[i].unsqueeze(0))
                        embedding_cache[image_path] = embedding.cpu() # Cache on CPU
                        image_embeddings.append(embedding)
            image_embeddings = torch.cat(image_embeddings, dim=0)

            batch_loss = 0
            
            # Loop through each image in the batch for batched character processing
            for i in range(len(images)):
                points_i = gt_points[i].to(device)
                masks_i = gt_masks[i].to(device)
                num_chars = len(points_i)

                if num_chars == 0:
                    continue

                # Batched prompt generation
                prompt_points = points_i.reshape(num_chars, 1, 2)
                prompt_labels = torch.ones(num_chars, device=device).reshape(num_chars, 1)

                # Single prompt encoder call
                sparse_emb, dense_emb = sam.prompt_encoder(
                    points=(prompt_points, prompt_labels), boxes=None, masks=None
                )

                # Single mask decoder call
                pred_masks, pred_iou = sam.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=True,
                ) # [Num_Chars, 3, 256, 256]

                # Batched dice score and best mask selection
                target_masks = masks_i.unsqueeze(1) # [Num_Chars, 1, 256, 256]
                if target_masks.max() > 1.0:
                    target_masks = target_masks / 255.0
                dice_scores = batched_dice_loss(pred_masks, target_masks.repeat(1, 3, 1, 1))
                best_mask_idx = torch.argmax(dice_scores, dim=1)
                best_pred_masks = torch.stack([pred_masks[j, best_mask_idx[j], :, :] for j in range(num_chars)])

                # Batched loss calculation
                loss = loss_fn(best_pred_masks.unsqueeze(1), target_masks)
                if torch.isnan(loss):
                    print("NaN loss detected!")
                    print(f"best_pred_masks min: {best_pred_masks.min()}, max: {best_pred_masks.max()}, mean: {best_pred_masks.mean()}")
                    print(f"target_masks min: {target_masks.min()}, max: {target_masks.max()}, mean: {target_masks.mean()}")
                batch_loss += loss

            if len(images) > 0:
                avg_loss = batch_loss / len(images)
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(sam.parameters(), 1.0)
                optimizer.step()
                epoch_loss += avg_loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_epoch_loss}")
        
        # Log metrics to wandb
        wandb.log({"epoch": epoch + 1, "loss": avg_epoch_loss})
        
        # Create and log comparison image
        # Use the last batch for visualization
        with torch.no_grad():
            sam.eval()
            
            # Get the first image from the last batch
            # Denormalize the image using SAM's pixel_mean and pixel_std
            image_tensor = images[0].cpu()  # [3, 1024, 1024]
            
            # Denormalize using SAM's preprocessing values
            pixel_mean = sam.pixel_mean.detach().clone().view(-1, 1, 1).cpu()
            pixel_std = sam.pixel_std.detach().clone().view(-1, 1, 1).cpu()
            
            # Denormalize: x = (x * std) + mean
            image_denorm = (image_tensor * pixel_std) + pixel_mean
            image_np = image_denorm.permute(1, 2, 0).numpy()
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
            
            original_pil = Image.fromarray(image_np)
            
            gt_masks_np = [m.cpu().numpy() for m in gt_masks[0]]
            gt_points_np = [p.cpu().numpy() for p in gt_points[0]]
            
            pred_masks_list = []
            # Re-run prediction for visualization (batched)
            points_i = gt_points[0].to(device)
            num_chars = len(points_i)
            if num_chars > 0:
                prompt_points = points_i.reshape(num_chars, 1, 2)
                prompt_labels = torch.ones(num_chars, device=device).reshape(num_chars, 1)

                sparse_emb, dense_emb = sam.prompt_encoder(
                    points=(prompt_points, prompt_labels), boxes=None, masks=None
                )

                pred_masks, _ = sam.mask_decoder(
                    image_embeddings=image_embeddings[0].unsqueeze(0),
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=True,
                )

                dice_scores = batched_dice_loss(pred_masks, gt_masks[0].to(device).unsqueeze(1).repeat(1, 3, 1, 1))
                best_mask_idx = torch.argmax(dice_scores, dim=1)
                best_pred_masks = torch.stack([pred_masks[j, best_mask_idx[j], :, :] for j in range(num_chars)])

                # Upscale the masks to the original image size
                upscaled_masks = sam.postprocess_masks(
                    best_pred_masks.unsqueeze(1),
                    input_size=images[0].shape[-2:],
                    original_size=(original_pil.height, original_pil.width)
                ).squeeze(1)

                for pred_mask in upscaled_masks:
                    pred_mask_np = (pred_mask > sam.mask_threshold).cpu().numpy()
                    pred_masks_list.append(pred_mask_np)

            # Transform GT masks from 256x256 to original image size for visualization
            gt_masks_upscaled = []
            for mask in gt_masks_np:
                # Resize from 256x256 to original image size
                mask_pil = Image.fromarray((mask).astype(np.uint8), 'L')
                mask_resized = mask_pil.resize((original_pil.width, original_pil.height), Image.NEAREST)
                gt_masks_upscaled.append(np.array(mask_resized) > 0.5)

            # Create overlays
            os.makedirs("comparison_images", exist_ok=True)
            gt_overlay_path = f"comparison_images/epoch_{epoch+1}_gt_overlay.png"
            pred_overlay_path = f"comparison_images/epoch_{epoch+1}_pred_overlay.png"
            comparison_path = f"comparison_images/epoch_{epoch+1}_comparison.png"
            
            gt_overlay = create_overlay_image(image_np, gt_masks_upscaled, gt_points_np, gt_overlay_path)
            pred_overlay = create_overlay_image(image_np, pred_masks_list, gt_points_np, pred_overlay_path)
            
            # Create and save comparison image
            create_comparison_image(original_pil, gt_overlay, pred_overlay, comparison_path)
            
            # Log images to wandb
            wandb.log({
                "comparison_images": wandb.Image(comparison_path, caption=f"Epoch {epoch+1} - Original | GT | Prediction"),
                # "gt_overlay": wandb.Image(gt_overlay_path, caption=f"Epoch {epoch+1} - Ground Truth"),
                # "pred_overlay": wandb.Image(pred_overlay_path, caption=f"Epoch {epoch+1} - Prediction"),
                "epoch": epoch + 1
            })

        # Save model checkpoint
        checkpoint_path = f"finetuned_models_v1/finetuned_sam_epoch_{epoch+1}.pth"
        os.makedirs("finetuned_models_v1", exist_ok=True)
        torch.save(sam.state_dict(), checkpoint_path)
        
        # Log checkpoint path to wandb
        wandb.log({"checkpoint_path": checkpoint_path, "epoch": epoch + 1})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='vit_b')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1) # Start with 1 due to memory
    parser.add_argument('--visualize', action='store_true', help="Generate and save comparison images.")
    args = parser.parse_args()
    main(args)
