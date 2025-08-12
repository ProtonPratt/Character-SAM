# prepare_data.py
import os
import json
from glob import glob

def create_master_index(data_root, output_file):
    """
    Scans the dataset directory and creates a single JSON index file.
    """
    master_list = []
    image_sets = sorted(glob(os.path.join(data_root, 'image_set_*')))

    print(f"Found {len(image_sets)} image sets. Processing...")

    for set_path in image_sets:
        annotations_path = os.path.join(set_path, 'annotations.json')
        image_path = os.path.join(set_path, 'image', 'generated_inscription.png')
        mask_dir = os.path.join(set_path, 'masks')

        if not os.path.exists(annotations_path):
            continue

        with open(annotations_path, 'r') as f:
            annotations = json.load(f)

        # Get all mask paths for this image
        mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')))

        # Create one entry per image
        image_entry = {
            'image_path': image_path,
            'mask_paths': mask_paths,
            'annotations': annotations # Keep annotations to get bboxes for points
        }
        master_list.append(image_entry)

    with open(output_file, 'w') as f:
        json.dump(master_list, f, indent=2)

    print(f"Master index created at {output_file}")

if __name__ == '__main__':
    create_master_index('/scratch/pratyush.jena/Characters_Inscriptions/kannada_char_generator/output_continuous_render', 'master_index.json')