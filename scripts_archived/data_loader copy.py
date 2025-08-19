# data_loader.py
import torch
import numpy as np
import json
from PIL import Image
from torch.nn import functional as F
from segment_anything.utils.transforms import ResizeLongestSide

class GuidedPromptDataset(torch.utils.data.Dataset):
    def __init__(self, index_file, sam_model):
        with open(index_file, 'r') as f:
            self.index = json.load(f)
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.sam_model = sam_model

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        record = self.index[idx]

        # Load image
        image = Image.open(record['image_path']).convert("RGB")
        original_size = image.size[::-1] # (H, W)

        # Preprocess image for SAM
        # Apply transform but ensure we don't exceed encoder size
        transformed_image = self.transform.apply_image(np.array(image))
        image_tensor = torch.as_tensor(transformed_image, device=self.sam_model.device)
        image_tensor = image_tensor.permute(2, 0, 1).contiguous()

        # Normalize
        pixel_mean = self.sam_model.pixel_mean
        pixel_std = self.sam_model.pixel_std
        x = (image_tensor - pixel_mean) / pixel_std

        h, w = x.shape[-2:]
        target_size = self.sam_model.image_encoder.img_size

        # Only pad, never crop
        padh = max(0, target_size - h)
        padw = max(0, target_size - w)

        # If image is larger than target, we need to handle it differently
        if h > target_size or w > target_size:
            # Option 1: Resize to fit exactly
            x = F.interpolate(x.unsqueeze(0), (target_size, target_size), mode='bilinear', align_corners=False).squeeze(0)
            padded_image = x
        else:
            # Normal padding case
            padded_image = F.pad(x, (0, padw, 0, padh))

        # --- Process masks and points ---
        gt_masks_list = []
        gt_points_list = []

        for i, char_annotation in enumerate(record['annotations']):
            # Load the corresponding mask
            mask = Image.open(record['mask_paths'][i])
            gt_mask_np = np.array(mask)
            
            # First apply the same transform as the image (preserve aspect ratio)
            transformed_mask = self.transform.apply_image(gt_mask_np)
            gt_mask_tensor = torch.as_tensor(transformed_mask, dtype=torch.float, device=self.sam_model.device)
            
            # If binary, ensure it stays binary after transformation
            if gt_mask_np.max() == 1:
                gt_mask_tensor = (gt_mask_tensor > 0.5).float()
            
            # Get dimensions of transformed mask
            mask_h, mask_w = gt_mask_tensor.shape
            
            # Apply the same padding/resizing logic as for the image
            if mask_h > target_size or mask_w > target_size:
                # Resize to fit exactly if larger than target size
                padded_mask = F.interpolate(
                    gt_mask_tensor.unsqueeze(0).unsqueeze(0), 
                    (target_size, target_size), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0).squeeze(0)
            else:
                # Pad with zeros if smaller than target size
                padded_mask = F.pad(
                    gt_mask_tensor.unsqueeze(0), 
                    (0, padw, 0, padh)
                ).squeeze(0)
            
            # Then downsample to 256x256 for loss calculation
            gt_mask_resized = F.interpolate(
                padded_mask.unsqueeze(0).unsqueeze(0),
                (256, 256),
                mode='bilinear',
                align_corners=False
            )
            gt_masks_list.append(gt_mask_resized.squeeze())

            # Create point prompt from bbox center
            bbox = char_annotation['bbox']
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            gt_points_list.append([x_center, y_center])

        # Transform points
        points_np = np.array(gt_points_list)
        transformed_points = self.transform.apply_coords(points_np, original_size)
        points_tensor = torch.as_tensor(transformed_points, device=self.sam_model.device)

        return {
            'image': padded_image,
            'gt_masks': torch.stack(gt_masks_list), # [Num_Chars, 256, 256]
            'gt_points': points_tensor # [Num_Chars, 2]
        }