# inference.py
import torch
import numpy as np
import json
import os
import argparse
from PIL import Image, ImageDraw
from tqdm import tqdm
import random

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

def calculate_iou(pred_mask, gt_mask):
    """Calculate IoU between predicted and ground truth masks."""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def save_comparison_image(original_np, gt_mask_np, pred_mask_np, point, output_path):
    """
    Saves a side-by-side comparison image for a single character.
    - Left: Original character crop with the prompt point.
    - Middle: Ground truth mask.
    - Right: Predicted mask.
    """
    # Check if any of the inputs are empty or invalid
    if original_np.size == 0 or gt_mask_np.size == 0 or pred_mask_np.size == 0:
        print(f"Warning: Empty crop detected, skipping comparison image: {output_path}")
        return
    
    # Check if dimensions match
    if original_np.shape[:2] != gt_mask_np.shape[:2] or original_np.shape[:2] != pred_mask_np.shape[:2]:
        print(f"Warning: Dimension mismatch in crops, skipping: {output_path}")
        return
    
    # Normalize masks to be 0-255 and convert to RGB for coloring
    gt_mask_viz = (gt_mask_np * 255).astype(np.uint8)
    gt_mask_viz = Image.fromarray(gt_mask_viz).convert("RGB")
    
    pred_mask_viz = (pred_mask_np * 255).astype(np.uint8)
    pred_mask_viz = Image.fromarray(pred_mask_viz).convert("RGB")
    
    # Create an image from the original crop and draw the prompt point
    try:
        original_viz = Image.fromarray(original_np)
        draw = ImageDraw.Draw(original_viz)
        
        # Draw a small circle at the prompt location
        radius = 5
        point_x, point_y = point
        
        # Ensure point is within image bounds
        img_width, img_height = original_viz.size
        if 0 <= point_x < img_width and 0 <= point_y < img_height:
            draw.ellipse(
                (point_x - radius, point_y - radius, point_x + radius, point_y + radius),
                fill='red',
                outline='white'
            )
        
        # Combine the three images horizontally
        width, height = original_viz.size
        comparison_img = Image.new('RGB', (width * 3, height))
        comparison_img.paste(original_viz, (0, 0))
        comparison_img.paste(gt_mask_viz, (width, 0))
        comparison_img.paste(pred_mask_viz, (width * 2, 0))
        
        comparison_img.save(output_path)
        
    except Exception as e:
        print(f"Error saving comparison image {output_path}: {e}")

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

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Model
    print("Loading model...")
    sam = sam_model_registry[args.model_type]()
    sam.load_state_dict(torch.load(args.checkpoint))
    sam.to(device)
    sam.eval() # Set model to evaluation mode

    # 2. Load the dataset index
    with open(args.index_file, 'r') as f:
        index = json.load(f)

    # 3. Setup Transform
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    
    # 4. Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "combined_gt"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "combined_pred"), exist_ok=True)
    
    # Initialize IoU tracking
    image_iou_data = []  # For average IoU per image
    character_iou_data = []  # For individual character IoU
    
    # 5. Inference Loop
    print(f"Running inference on {len(index)} images...")
    for i, record in enumerate(tqdm(index, desc="Processing images")):
        image_name = os.path.splitext(os.path.basename(record['image_path']))[0]
        
        # Load the full inscription image
        image_np = np.array(Image.open(record['image_path']).convert("RGB"))
        original_size = image_np.shape[:2]
        print(f"Processing image: {image_name} (size: {original_size})")

        # --- Preprocess Image (same as in data_loader.py) ---
        input_image = sam.preprocess(
            torch.as_tensor(transform.apply_image(image_np), device=device)
                 .permute(2, 0, 1)
                 .contiguous()
                 .unsqueeze(0)
        )
        
        # --- Pre-compute Image Embedding ---
        with torch.no_grad():
            image_embedding = sam.image_encoder(input_image)

        # Lists to store all masks and points for this image
        gt_masks_list = []
        pred_masks_list = []
        points_list = []
        image_ious = []  # Store IoUs for this image

        # --- Process each character in the image ---
        for char_idx, char_annotation in enumerate(record['annotations']):
            try:
                # Load the ground truth mask
                gt_mask_path = record['mask_paths'][char_idx]
                gt_mask_np = np.array(Image.open(gt_mask_path)) > 0

                # Get the prompt point from the character's bounding box
                bbox = char_annotation['bbox']
                point_coords_orig = np.array([[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]])
                point_labels = np.array([1])

                # --- Transform the prompt point ---
                transformed_points = transform.apply_coords(point_coords_orig, original_size)
                point_coords_torch = torch.as_tensor(transformed_points, dtype=torch.float, device=device)
                point_labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)

                # Format for the model: [Batch, N_points, Coords]
                point_prompt = (point_coords_torch[None, ...], point_labels_torch[None, ...])
                
                # --- Predict the mask ---
                with torch.no_grad():
                    # Get sparse embeddings from the prompt
                    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                        points=point_prompt, boxes=None, masks=None
                    )
                    
                    # Get mask predictions
                    low_res_masks, iou_predictions = sam.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False, # We want the single best mask for this prompt
                    )

                    # Upscale the mask to the original image size
                    in_resize = transform.get_preprocess_shape(original_size[0], original_size[1], 1024)
                    final_masks = sam.postprocess_masks(low_res_masks, in_resize, original_size)
                    
                    # Binarize the prediction
                    predicted_mask_np = (final_masks > sam.mask_threshold).squeeze().cpu().numpy()
                    
                    # Calculate IoU
                    char_iou = calculate_iou(predicted_mask_np, gt_mask_np)
                    image_ious.append(char_iou)
                    
                    # Store character-level IoU data
                    character_iou_data.append({
                        "image_name": image_name,
                        "image_path": record['image_path'],
                        "character_index": char_idx,
                        "character": char_annotation.get('character', 'unknown'),
                        "bbox": bbox,
                        "iou": float(char_iou),
                        "predicted_iou": float(iou_predictions.item())
                    })

                # Store masks and points
                gt_masks_list.append(gt_mask_np)
                pred_masks_list.append(predicted_mask_np)
                points_list.append(point_coords_orig[0])

                # --- Save the comparison image ---
                # Crop the original image and masks to the character's bounding box for a clean view
                x1, y1, x2, y2 = [int(c) for c in bbox]
                
                # Validate bounding box
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image_np.shape[1], x2)
                y2 = min(image_np.shape[0], y2)
                
                # Check if bounding box is valid
                if x2 <= x1 or y2 <= y1:
                    print(f"Warning: Invalid bounding box for character {char_idx}, skipping comparison image")
                    continue
                
                # Add a small margin for better visualization
                margin = 10
                crop_x1 = max(0, x1 - margin)
                crop_y1 = max(0, y1 - margin)
                crop_x2 = min(image_np.shape[1], x2 + margin)
                crop_y2 = min(image_np.shape[0], y2 + margin)
                
                # Ensure crop dimensions are valid
                if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                    print(f"Warning: Invalid crop dimensions for character {char_idx}, skipping comparison image")
                    continue
                
                # Crop original image, GT mask, and predicted mask
                original_crop = image_np[crop_y1:crop_y2, crop_x1:crop_x2]
                gt_mask_crop = gt_mask_np[crop_y1:crop_y2, crop_x1:crop_x2]
                pred_mask_crop = predicted_mask_np[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Adjust the prompt point to the cropped coordinate system
                cropped_point = (point_coords_orig[0, 0] - crop_x1, point_coords_orig[0, 1] - crop_y1)
                
                # Define output path
                # Create a folder for this image
                image_output_dir = os.path.join(args.output_dir, f"image_{i:03d}_{image_name}")
                os.makedirs(image_output_dir, exist_ok=True)
                
                output_filename = f"{image_name}_char_{char_idx:03d}.png"
                output_path = os.path.join(image_output_dir, output_filename)
                
                # Save the visual comparison
                save_comparison_image(original_crop, gt_mask_crop, pred_mask_crop, cropped_point, output_path)
                
                # --- Save full size masks ---
                # Create mask output directories
                gt_mask_dir = os.path.join(args.output_dir, "full_masks_gt")
                pred_mask_dir = os.path.join(args.output_dir, "full_masks_pred")
                os.makedirs(gt_mask_dir, exist_ok=True)
                os.makedirs(pred_mask_dir, exist_ok=True)
                
                # Save GT mask
                gt_mask_filename = f"{image_name}_char_{i}_{char_idx:03d}_gt.png"
                gt_mask_output_path = os.path.join(gt_mask_dir, gt_mask_filename)
                Image.fromarray((gt_mask_np * 255).astype(np.uint8)).save(gt_mask_output_path)
                
                # Save predicted mask
                pred_mask_filename = f"{image_name}_char_{i}_{char_idx:03d}_pred.png"
                pred_mask_output_path = os.path.join(pred_mask_dir, pred_mask_filename)
                Image.fromarray((predicted_mask_np * 255).astype(np.uint8)).save(pred_mask_output_path)
                
            except Exception as e:
                print(f"Error processing character {char_idx} in image {image_name}: {e}")
                continue  # Skip this character and continue with the next one

        # Calculate average IoU for this image
        avg_iou = np.mean(image_ious) if image_ious else 0.0
        image_iou_data.append({
            "image_name": image_name,
            "image_path": record['image_path'],
            "num_characters": len(image_ious),
            "average_iou": float(avg_iou),
            "min_iou": float(np.min(image_ious)) if image_ious else 0.0,
            "max_iou": float(np.max(image_ious)) if image_ious else 0.0,
            "std_iou": float(np.std(image_ious)) if image_ious else 0.0
        })

        # --- Create combined overlay images ---
        # Set random seed for consistent colors between GT and pred images
        random.seed(42 + i)  # Different seed for each image but consistent between GT/pred
        
        # GT overlay
        gt_output_path = os.path.join(args.output_dir, "combined_gt", f"{image_name}_gt_overlay_{i}.png")
        create_overlay_image(image_np, gt_masks_list, points_list, gt_output_path, "GT")
        
        # Reset seed for same colors
        random.seed(42 + i)
        
        # Prediction overlay
        pred_output_path = os.path.join(args.output_dir, "combined_pred", f"{image_name}_pred_overlay_{i}.png")
        create_overlay_image(image_np, pred_masks_list, points_list, pred_output_path, "Pred")
        
        print(f"Saved combined overlays for {image_name} (Avg IoU: {avg_iou:.4f})")
        
        # Optional: limit number of images for testing
        if i >= 10:  # Process only first 10 images
            break

    # --- Save IoU metrics to JSON files ---
    # Save image-level IoU data
    image_iou_file = os.path.join(args.output_dir, "image_iou_metrics.json")
    with open(image_iou_file, 'w') as f:
        json.dump({
            "summary": {
                "total_images": len(image_iou_data),
                "overall_avg_iou": float(np.mean([img["average_iou"] for img in image_iou_data])),
                "overall_std_iou": float(np.std([img["average_iou"] for img in image_iou_data]))
            },
            "per_image_metrics": image_iou_data
        }, f, indent=2)
    
    # Save character-level IoU data
    character_iou_file = os.path.join(args.output_dir, "character_iou_metrics.json")
    with open(character_iou_file, 'w') as f:
        json.dump({
            "summary": {
                "total_characters": len(character_iou_data),
                "overall_avg_iou": float(np.mean([char["iou"] for char in character_iou_data])),
                "overall_std_iou": float(np.std([char["iou"] for char in character_iou_data]))
            },
            "per_character_metrics": character_iou_data
        }, f, indent=2)

    print("✅ Combined overlay generation completed!")
    print(f"📊 IoU metrics saved to:")
    print(f"   - Image-level: {image_iou_file}")
    print(f"   - Character-level: {character_iou_file}")
    
    # Print summary statistics
    overall_avg = np.mean([char["iou"] for char in character_iou_data])
    print(f"📈 Overall average IoU: {overall_avg:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference with a finetuned SAM model.")
    parser.add_argument('--model_type', type=str, default='vit_b', help="The type of SAM model.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to your finetuned model checkpoint.")
    parser.add_argument('--index_file', type=str, default='master_index.json', help="Path to the dataset index JSON file.")
    parser.add_argument('--output_dir', type=str, default='inference_outputs_van', help="Directory to save the visual results.")
    args = parser.parse_args()
    main(args)