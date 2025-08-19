# test_data_loader.py
import torch
import time
import argparse
from tqdm import tqdm
import pickle
import tracemalloc

from data_loader import GuidedPromptDataset
from train import custom_collate_fn # Import from train.py

def main(args):
    print("Initializing dataset...")
    # We need a dummy model to initialize the dataset, but it won't be used for training
    # So we can load a small model to save memory
    sam = sam_model_registry['vit_b'](checkpoint=None)
    dataset = GuidedPromptDataset(index_file='master_index.json', sam_model=sam)
    
    print("Creating DataLoader...")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No need to shuffle for this test
        collate_fn=custom_collate_fn,
        num_workers=12
    )

    print(f"Loading all {len(dataset)} samples from the dataset...")
    
    tracemalloc.start()
    start_time = time.time()

    loaded_data = []
    for batch in tqdm(data_loader, desc="Loading data"):
        loaded_data.append(batch)

    end_time = time.time()
    total_time = end_time - start_time
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nSuccessfully loaded all data in {total_time:.2f} seconds.")
    print(f"Current memory usage: {current / 10**6:.2f}MB")
    print(f"Peak memory usage: {peak / 10**6:.2f}MB")

    # # Save the loaded data to a pickle file
    # print("Saving loaded data to 'loaded_data.pickle'...")
    # with open('loaded_data.pickle', 'wb') as f:
    #     pickle.dump(loaded_data, f)
    
    print("✅ Data loading test complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test the data loading time for GuidedPromptDataset.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for the DataLoader.")
    args = parser.parse_args()
    
    # We need to import sam_model_registry to run this script
    from segment_anything import sam_model_registry
    
    main(args)