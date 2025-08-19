# train_cached.py
import torch
import argparse
from tqdm import tqdm
from segment_anything import sam_model_registry
import numpy as np
from PIL import Image, ImageDraw
import random

from data_loader import GuidedPromptDataset
from utils import CombinedLoss, dice_loss
import os

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
        
        # Create colored mask
        mask_rgba = Image.new('RGBA', overlay_img.size, (0, 0, 0, 0))
        
        # Convert mask to RGBA with the random color
        mask_colored = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
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
    
    for sample in batch:
        images.append(sample['image'])
        gt_masks.append(sample['gt_masks'])  # Keep as list - don't stack
        gt_points.append(sample['gt_points'])  # Keep as list - don't stack
        image_paths.append(sample['image_path'])
    
    # Only stack images since they have fixed dimensions
    images = torch.stack(images, dim=0)
    
    return {
        'image': images,
        'gt_masks': gt_masks,  # List of tensors with different sizes
        'gt_points': gt_points,  # List of tensors with different sizes
        'image_path': image_paths
    }

def main(args):
    device = "cuda:2" if torch.cuda.is_available() else "cpu"

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
        index_file='master_index.json', 
        img_size=sam.image_encoder.img_size, 
        pixel_mean=sam.pixel_mean, 
        pixel_std=sam.pixel_std,
        return_path=True
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
                    image_embeddings.append(embedding_cache[image_path])
                else:
                    with torch.no_grad():
                        embedding = sam.image_encoder(images[i].unsqueeze(0))
                        embedding_cache[image_path] = embedding.cpu() # Cache on CPU
                        image_embeddings.append(embedding)
            image_embeddings = torch.cat(image_embeddings, dim=0).to(device)

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
        
        if args.visualize:
            # Create and log comparison image
            # Use the last batch for visualization
            with torch.no_grad():
                sam.eval()
                
                # Get the first image from the last batch
                image_np = images[0].permute(1, 2, 0).cpu().numpy() * 255
                image_np = image_np.astype(np.uint8)
                
                gt_masks_np = [m.cpu().numpy() for m in gt_masks[0]]
                gt_points_np = [p.cpu().numpy() for p in gt_points[0]]
                
                pred_masks_list = []
                for j in range(len(gt_points[0])):
                    prompt_point = gt_points[0][j].unsqueeze(0).unsqueeze(0)
                    prompt_label = torch.tensor([[1]], device=device)
                    
                    sparse_emb, dense_emb = sam.prompt_encoder(
                        points=(prompt_point, prompt_label), boxes=None, masks=None
                    )
                    
                    pred_masks, _ = sam.mask_decoder(
                        image_embeddings=image_embeddings[0].unsqueeze(0),
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=True,
                    )
                    
                    # Find the best mask
                    dice_scores = torch.stack([1 - dice_loss(m, gt_masks[0][j].unsqueeze(0).unsqueeze(0)) for m in pred_masks.squeeze(0)])
                    best_mask_idx = torch.argmax(dice_scores)
                    best_pred_mask = pred_masks[:, best_mask_idx, :, :].unsqueeze(1)
                    
                    best_pred_mask_np = sam.postprocess_masks(
                        best_pred_mask,
                        input_size=images[0].shape[-2:],
                        original_size=images[0].shape[-2:]
                    )
                    best_pred_mask_np = (best_pred_mask_np > sam.mask_threshold).squeeze().cpu().numpy()
                    pred_masks_list.append(best_pred_mask_np)

                # Create overlays
                os.makedirs("comparison_images", exist_ok=True)
                gt_overlay_path = f"comparison_images/epoch_{epoch+1}_gt_overlay.png"
                pred_overlay_path = f"comparison_images/epoch_{epoch+1}_pred_overlay.png"
                
                gt_overlay = create_overlay_image(image_np, gt_masks_np, gt_points_np, gt_overlay_path)
                pred_overlay = create_overlay_image(image_np, pred_masks_list, gt_points_np, pred_overlay_path)
                
                # Create and save comparison image
                comparison_path = f"comparison_images/epoch_{epoch+1}_comparison.png"
                original_pil = Image.fromarray(image_np)
                create_comparison_image(original_pil, gt_overlay, pred_overlay, comparison_path)

        # Save model checkpoint
        checkpoint_path = f"finetuned_models/finetuned_sam_epoch_{epoch+1}.pth"
        os.makedirs("finetuned_models", exist_ok=True)
        torch.save(sam.state_dict(), checkpoint_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='vit_b')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1) # Start with 1 due to memory
    parser.add_argument('--visualize', action='store_true', help="Generate and save comparison images.")
    args = parser.parse_args()
    main(args)