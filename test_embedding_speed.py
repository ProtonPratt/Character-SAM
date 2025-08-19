# test_embedding_speed.py
import torch
import time
import argparse
from tqdm import tqdm

from data_loader import GuidedPromptDataset
from segment_anything import sam_model_registry

def main(args):
    device = "cuda:2" if torch.cuda.is_available() else "cpu"

    # 1. Setup Model
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device)
    sam.eval()

    # 2. Setup Dataloader
    dataset = GuidedPromptDataset(
        index_file='master_index.json', 
        img_size=sam.image_encoder.img_size, 
        pixel_mean=sam.pixel_mean, 
        pixel_std=sam.pixel_std,
        return_path=True
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=12
    )

    # --- Time uncached embedding calculation ---
    print("\n--- Testing uncached embedding speed ---")
    start_time_uncached = time.time()
    embedding_cache = {}

    for batch in tqdm(data_loader, desc="Calculating embeddings (uncached)"):
        images = batch['image'].to(device)
        image_paths = batch['image_path']

        with torch.no_grad():
            embeddings = sam.image_encoder(images)
        
        # Store in cache for the next test
        for i, image_path in enumerate(image_paths):
            embedding_cache[image_path] = embeddings[i].cpu()

    end_time_uncached = time.time()
    total_time_uncached = end_time_uncached - start_time_uncached
    print(f"Uncached embedding calculation took: {total_time_uncached:.2f} seconds")

    # --- Time cached embedding retrieval ---
    print("\n--- Testing cached embedding speed ---")
    start_time_cached = time.time()

    for batch in tqdm(data_loader, desc="Retrieving embeddings (cached)"):
        image_paths = batch['image_path']

        # Retrieve from cache
        retrieved_embeddings = []
        for image_path in image_paths:
            retrieved_embeddings.append(embedding_cache[image_path].to(device))
        
        retrieved_embeddings = torch.stack(retrieved_embeddings, dim=0)

    end_time_cached = time.time()
    total_time_cached = end_time_cached - start_time_cached
    print(f"Cached embedding retrieval took: {total_time_cached:.2f} seconds")

    # --- Summary ---
    print("\n--- Summary ---")
    print(f"Uncached time: {total_time_uncached:.2f}s")
    print(f"Cached time:   {total_time_cached:.2f}s")
    if total_time_cached > 0:
        speedup = total_time_uncached / total_time_cached
        print(f"Speed-up:      {speedup:.2f}x")
    print("✅ Speed test complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test the speed of image embedding calculation.")
    parser.add_argument('--model_type', type=str, default='vit_b', help="The type of SAM model.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the SAM model checkpoint.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for the DataLoader.")
    args = parser.parse_args()
    main(args)
