# train.py
import torch
import argparse
from tqdm import tqdm
from segment_anything import sam_model_registry
from PIL import Image
import numpy as np

from data_loader import GuidedPromptDataset
from utils import CombinedLoss, dice_loss
import os

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
    device = "cuda:2" if torch.cuda.is_available() else "cpu"

    # 1. Setup Model
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device)
    sam.eval()  # Set to evaluation mode

    # 3. Setup Dataloader with custom collate function
    dataset = GuidedPromptDataset(index_file='master_index.json', sam_model=sam)
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # No need to shuffle for inference
        collate_fn=custom_collate_fn
    )

    # 4. Inference Loop (No Training)
    print("Starting mask generation (inference only)...")
    
    with torch.no_grad():  # No gradients needed for inference
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            images = batch['image'] # [B, 3, 1024, 1024]
            gt_masks = batch['gt_masks'] # List of [Num_Chars, 256, 256]
            gt_points = batch['gt_points'] # List of [Num_Chars, 2]
            
            # Get image embeddings
            image_embeddings = sam.image_encoder(images)
            
            # Loop through each image in the batch
            for i in range(len(images)):
                points_i = gt_points[i]
                masks_i = gt_masks[i]
                
                # Loop through each character (prompt) in the image
                for j in range(len(points_i)):
                    prompt_point = points_i[j].unsqueeze(0).unsqueeze(0)
                    prompt_label = torch.tensor([[1]], device=device) # Foreground point
                    target_mask = masks_i[j].unsqueeze(0).unsqueeze(0) # [1, 1, 256, 256]
                    
                    # Ensure target_mask is in [0,1] range
                    if target_mask.max() > 1.0:
                        target_mask = target_mask / 255.0

                    # Get prompt embeddings
                    sparse_emb, dense_emb = sam.prompt_encoder(
                        points=(prompt_point, prompt_label), boxes=None, masks=None
                    )

                    # Generate predicted masks
                    pred_masks, pred_iou = sam.mask_decoder(
                        image_embeddings=image_embeddings[i].unsqueeze(0),
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=True,
                    ) # [1, 3, 256, 256]

                    # Find the best of the 3 predicted masks
                    dice_scores = [dice_loss(m, target_mask) for m in pred_masks.squeeze(0)]
                    best_mask_idx = torch.argmin(torch.tensor(dice_scores))
                    best_pred_mask = pred_masks[:, best_mask_idx, :, :].unsqueeze(1)
                    
                    # Save predicted and ground truth masks
                    os.makedirs("generated_masks", exist_ok=True)
                    os.makedirs("generated_masks/predicted", exist_ok=True)
                    os.makedirs("generated_masks/ground_truth", exist_ok=True)
                    
                    # Save predicted mask
                    pred_mask_np = (best_pred_mask.squeeze().cpu().detach().numpy() * 255).astype(np.uint8)
                    pred_img = Image.fromarray(pred_mask_np)
                    pred_img.save(f"generated_masks/predicted/batch_{batch_idx}_img_{i}_char_{j}.png")
                    
                    # Save ground truth mask
                    gt_mask_np = (target_mask.squeeze().cpu().detach().numpy() * 255).astype(np.uint8)
                    gt_img = Image.fromarray(gt_mask_np)
                    gt_img.save(f"generated_masks/ground_truth/batch_{batch_idx}_img_{i}_char_{j}.png")
                    
                    print(f"Saved masks for batch {batch_idx}, image {i}, character {j}")
            
            # Optional: limit number of batches for testing
            if batch_idx >= 10:  # Process only first 10 batches
                print(f"Processed {batch_idx + 1} batches. Stopping.")
                break
    
    print("✅ Mask generation completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='vit_h')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    main(args)