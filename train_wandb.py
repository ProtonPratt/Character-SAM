# train.py
import torch
import argparse
from tqdm import tqdm
from segment_anything import sam_model_registry
import numpy as np
from PIL import Image, ImageDraw
import wandb
import random

from data_loader import GuidedPromptDataset
from utils import CombinedLoss, dice_loss
import os
import matplotlib.pyplot as plt

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
    parser.add_argument('--index_file', type=str, default='master_index.json', help='Path to the JSON index file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1) # Start with 1 due to memory
    parser.add_argument('--comparison_images_dir', type=str, default='comparison_images_250', help='Directory to save comparison images')
    parser.add_argument('--checkpoint_dir', type=str, default='finetuned_models_250', help='Directory to save model checkpoints')
    args = parser.parse_args()
    main(args)