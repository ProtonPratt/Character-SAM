# visualize_dataset.py
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def _sam_device(sam_model):
    # SAM models don't expose `.device` directly; infer it from parameters.
    return next(sam_model.parameters()).device

def denorm_to_uint8(img_chw, sam_model):
    """
    img_chw: (3, H, W) tensor that was normalized with sam_model.pixel_mean/std
    returns: (H, W, 3) uint8 numpy for display
    """
    device = img_chw.device
    mean = sam_model.pixel_mean.to(device)        # (3,1,1)
    std  = sam_model.pixel_std.to(device)         # (3,1,1)
    x = (img_chw * std) + mean
    x = x.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return x

def tile_masks(masks_nhw, max_cols=6):
    """
    masks_nhw: (N, 256, 256) tensor/np with values ~0..255 or 0..1
    returns: grid (H, W) float32 in [0,1], rows, cols
    """
    m = masks_nhw.detach().cpu().float().numpy()
    # Normalize to [0,1] for display
    if m.max() > 1.0:
        m = m / 255.0
    n = m.shape[0]
    if n == 0:
        return np.zeros((256, 256), dtype=np.float32), 1, 1
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    grid = np.zeros((rows * 256, cols * 256), dtype=np.float32)
    for i in range(n):
        r, c = divmod(i, cols)
        grid[r*256:(r+1)*256, c*256:(c+1)*256] = m[i]
    return grid, rows, cols

def show_sample(sample, sam_model, title=None):
    """
    sample: dict returned by your dataset.__getitem__
      - image: (3,H,W) normalized & padded
      - gt_points: (N,2) xy in transformed coords (before padding; padding added on right/bottom only, so OK)
      - gt_masks: (N,256,256)
    """
    img = denorm_to_uint8(sample['image'], sam_model)
    pts = sample['gt_points'].detach().cpu().numpy() if len(sample['gt_points']) else np.zeros((0,2))
    masks = sample['gt_masks']  # (N,256,256)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # --- Left: image + points ---
    axes[0].imshow(img)
    if len(pts):
        axes[0].scatter(pts[:,0], pts[:,1], s=18, linewidths=0.6, edgecolors='k')
    axes[0].set_title('Transformed + padded image with point prompts')
    axes[0].axis('off')

    # --- Right: tiled masks ---
    grid, rows, cols = tile_masks(masks, max_cols=6)
    axes[1].imshow(grid, cmap='gray', vmin=0, vmax=1)
    # optional: index labels per tile
    for i in range(len(masks)):
        r, c = divmod(i, cols)
        y = r*256 + 12
        x = c*256 + 8
        axes[1].text(x, y, str(i), fontsize=9, color='white', ha='left', va='top')
    axes[1].set_title(f'GT masks (N={len(masks)}) at 256×256')
    axes[1].axis('off')

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# -------------------- Example usage --------------------
if __name__ == "__main__":
    import json
    from segment_anything import sam_model_registry
    from data_loader import GuidedPromptDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry['vit_b']('./checkpoints/sam_vit_b_01ec64.pth')
    sam.to(device)

    dataset = GuidedPromptDataset(index_file='master_index.json', sam_model=sam)

    # visualize a few random samples
    for _ in range(3):
        idx = random.randrange(len(dataset))
        sample = dataset[idx]   # returns a dict (not batched)
        show_sample(sample, sam, title=f"sample #{idx}")
