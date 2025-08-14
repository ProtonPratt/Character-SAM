# train.py
import torch
import argparse
from tqdm import tqdm
from segment_anything import sam_model_registry
import numpy as np
from PIL import Image

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
    dataset = GuidedPromptDataset(index_file='master_index.json', sam_model=sam)
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=custom_collate_fn  # Add custom collate function
    )

    # 4. Training Loop
    for epoch in range(args.epochs):
        sam.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader):
            images = batch['image'] # [B, 3, 1024, 1024]
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
                    prompt_point = points_i[j].unsqueeze(0).unsqueeze(0)
                    prompt_label = torch.tensor([[1]], device=device) # Foreground point
                    target_mask = masks_i[j].unsqueeze(0).unsqueeze(0) # [1, 1, 256, 256]
                    
                    # save the target mask for visualization
                    os.makedirs("generated_masks", exist_ok=True)
                    os.makedirs("generated_masks/ground_truth_tar", exist_ok=True)
                    target_mask_np = (target_mask.squeeze().cpu().detach().numpy()).astype(np.uint8)
                    target_mask_img = Image.fromarray(target_mask_np)
                    # print(target_mask.max(), target_mask.min())
                    target_mask_img.save(f"generated_masks/ground_truth_tar/epoch_{epoch+1}_batch_{i}_char_{j}.png")
                    
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
                    # Note: You could also use pred_iou here, but direct dice is a strong signal
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
                    
                    # Save predicted and ground truth masks
                    os.makedirs("generated_masks", exist_ok=True)
                    os.makedirs("generated_masks/predicted", exist_ok=True)
                    os.makedirs("generated_masks/ground_truth", exist_ok=True)
                    
                    # # Save predicted mask
                    # pred_mask_np = (best_save_pred_mask.squeeze().cpu().detach().numpy() * 255).astype(np.uint8)
                    # pred_img = Image.fromarray(pred_mask_np)
                    # pred_img.save(f"generated_masks/predicted/epoch_{epoch+1}_batch_{i}_char_{j}.png")
                    
                    # # Save ground truth mask
                    # gt_mask_np = (target_mask.squeeze().cpu().detach().numpy() * 255).astype(np.uint8)
                    # gt_img = Image.fromarray(gt_mask_np)
                    # gt_img.save(f"generated_masks/ground_truth/epoch_{epoch+1}_batch_{i}_char_{j}.png")
                    
                    loss = loss_fn(best_pred_mask, target_mask)
                    batch_loss += loss

            if num_chars_in_batch > 0:
                avg_loss = batch_loss / num_chars_in_batch
                avg_loss.backward()
                optimizer.step()
                epoch_loss += avg_loss.item()
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/len(train_loader)}")
        
        # Save model checkpoint
        os.makedirs("finetuned_models", exist_ok=True)
        torch.save(sam.state_dict(), f"finetuned_models/finetuned_sam_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='vit_b')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1) # Start with 1 due to memory
    args = parser.parse_args()
    main(args)