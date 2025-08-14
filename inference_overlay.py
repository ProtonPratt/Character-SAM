# inference_visualize.py
import torch
import numpy as np
import argparse
import os
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn import functional as F

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.amg import (
    batched_mask_to_box,
    calculate_stability_score,
    build_point_grid,
    # We will need a mask-based NMS function
)

# You can place this helper function here
def overlay_masks_on_image(image_np, masks, alpha=0.5):
    # (Implementation from above)
    import cv2
    overlay = image_np.copy()
    for mask in masks:
        binary_mask = mask.astype(bool)
        color = np.random.randint(0, 256, 3, dtype=np.uint8)
        if np.any(binary_mask): # only apply if mask is not empty
            overlay[binary_mask] = cv2.addWeighted(
                overlay[binary_mask], 1 - alpha, color.astype(np.uint8), alpha, 0
            )
    return overlay

def mask_postprocessing(masks: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
    """Upscales masks to the original image size."""
    masks = F.interpolate(
        masks,
        (orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )
    return masks

def non_max_suppression_masks(masks, scores, iou_threshold=0.7):
    """
    Performs Non-Maximal Suppression on masks.
    
    Args:
        masks (torch.Tensor): A tensor of masks (N, H, W).
        scores (torch.Tensor): A tensor of scores for each mask (N,).
        iou_threshold (float): The IoU threshold for suppression.

    Returns:
        torch.Tensor: A tensor of indices of the masks to keep.
    """
    if len(masks) == 0:
        return torch.tensor([], dtype=torch.long)
        
    # Sort masks by score in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    
    keep = []
    while len(sorted_indices) > 0:
        current_idx = sorted_indices[0]
        keep.append(current_idx)
        
        if len(sorted_indices) == 1:
            break
            
        current_mask = masks[current_idx]
        other_indices = sorted_indices[1:]
        other_masks = masks[other_indices]
        
        # Calculate IoU between the current mask and all others
        intersection = torch.logical_and(current_mask, other_masks).sum(dim=(1, 2))
        union = torch.logical_or(current_mask, other_masks).sum(dim=(1, 2))
        iou = intersection / union
        
        # Keep masks with IoU below the threshold
        non_overlapping_indices = other_indices[iou <= iou_threshold]
        sorted_indices = non_overlapping_indices
        
    return torch.tensor(keep, dtype=torch.long)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Finetuned Model
    sam = sam_model_registry[args.model_type]()
    sam.load_state_dict(torch.load(args.finetuned_checkpoint))
    sam.to(device)
    sam.eval()

    # 2. Load Image and GT Data from Index
    with open('master_index.json', 'r') as f:
        index = json.load(f)
    
    record = index[args.image_idx]
    image_path = record['image_path']
    gt_mask_paths = record['mask_paths']
    
    original_image = Image.open(image_path).convert("RGB")
    original_image_np = np.array(original_image)
    orig_h, orig_w, _ = original_image_np.shape

    # 3. Preprocess Image for SAM
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    input_image = sam.preprocess(
        torch.as_tensor(transform.apply_image(original_image_np), device=device)
        .permute(2, 0, 1)
        .contiguous()
        .unsqueeze(0)
    )
    input_size = tuple(input_image.shape[-2:])

    # --- 4. "Segment Everything" Pipeline ---
    print("Running 'Segment Everything' pipeline...")
    with torch.no_grad():
        # a. Get image embedding
        image_embedding = sam.image_encoder(input_image)

        # b. Generate grid prompts
        points_per_side = 32
        point_grid = build_point_grid(points_per_side)
        
        # Add a batch dimension and labels
        grid_points = torch.from_numpy(point_grid).unsqueeze(0).to(device)
        grid_labels = torch.ones(grid_points.shape[:2], dtype=torch.int, device=device)

        # c. Run prompt and mask encoders
        sparse_emb, dense_emb = sam.prompt_encoder(
            points=(grid_points, grid_labels), boxes=None, masks=None
        )
        
        low_res_masks, iou_predictions = sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True,
        ) # Shape: [1, N_points*3, 256, 256]

        # d. Upscale masks and filter
        # Upscale to the padded input size, not original size yet
        upscaled_masks = sam.postprocess_masks(low_res_masks, input_size, input_size)
        
        # e. Calculate stability scores and get binary masks
        stability_scores = calculate_stability_score(
            upscaled_masks, sam.mask_threshold, 0.95 # Stability threshold
        )
        binary_masks = (upscaled_masks > sam.mask_threshold)

        # Squeeze batch dimension
        binary_masks = binary_masks.squeeze(0) # [N_masks, H, W]
        iou_predictions = iou_predictions.squeeze(0) # [N_masks, 3] -> use max
        stability_scores = stability_scores.squeeze(0)
        
        # Combine scores (you can tune this)
        if iou_predictions.dim() > 1:
            final_scores = stability_scores * iou_predictions.max(dim=1).values
        else:
            final_scores = stability_scores * iou_predictions
        
        # f. Non-Maximal Suppression (NMS)
        keep_indices = non_max_suppression_masks(binary_masks, final_scores, iou_threshold=0.7)
        final_masks_tensor = binary_masks[keep_indices]
        
        # g. Postprocess final masks to original image size
        final_masks_processed = sam.postprocess_masks(
            final_masks_tensor.unsqueeze(1).float(), input_size, (orig_h, orig_w)
        ).squeeze(1) > 0.5

        final_predicted_masks = final_masks_processed.cpu().numpy()
        print(f"Found {len(final_predicted_masks)} characters.")

    # --- 5. Prepare Ground Truth Masks ---
    print("Loading ground truth masks...")
    gt_masks = []
    for mask_path in gt_mask_paths:
        mask = np.array(Image.open(mask_path))
        # Ensure it's binary and has the same shape as the original image
        mask = (mask > 0).astype(np.uint8)
        gt_masks.append(mask)

    # --- 6. Create and Save Visualization ---
    print("Creating visualizations...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create prediction overlay
    pred_overlay = overlay_masks_on_image(original_image_np, final_predicted_masks)
    pred_img = Image.fromarray(pred_overlay)
    pred_save_path = os.path.join(args.output_dir, f"prediction_overlay_{args.image_idx}.png")
    pred_img.save(pred_save_path)
    print(f"Saved prediction overlay to {pred_save_path}")
    
    # Create ground truth overlay
    gt_overlay = overlay_masks_on_image(original_image_np, gt_masks)
    gt_img = Image.fromarray(gt_overlay)
    gt_save_path = os.path.join(args.output_dir, f"gt_overlay_{args.image_idx}.png")
    gt_img.save(gt_save_path)
    print(f"Saved ground truth overlay to {gt_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize finetuned SAM performance.")
    parser.add_argument('--model_type', type=str, default='vit_b')
    parser.add_argument('--finetuned_checkpoint', type=str, required=True, help="Path to your finetuned SAM model.")
    parser.add_argument('--image_idx', type=int, default=0, help="Index of the image from master_index.json to process.")
    parser.add_argument('--output_dir', type=str, default='outputs', help="Directory to save the visualization images.")
    args = parser.parse_args()
    main(args)