import gradio as gr
import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import numpy as np

# Load the SAM model
model_type = "vit_h"
model_path = "./checkpoints/sam_vit_h_4b8939.pth"  # Update this path
sam = sam_model_registry[model_type](checkpoint=model_path)
sam.to("cuda")  # Use "cpu" if no GPU is available
predictor = SamPredictor(sam)

# Function to process the image and predict segmentation
def segment_image(image, x, y):
    input_image = np.array(image)
    predictor.set_image(input_image)

    points = np.array([[x, y]])
    labels = np.array([1])  # Foreground label
    masks, _, _ = predictor.predict(point_coords=points, point_labels=labels, multimask_output=False)

    # Convert mask to image format
    mask_image = (masks[0] * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_image).convert("RGB")

    return mask_pil

# Gradio Interface
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Segment Anything Model Demo")
        
        with gr.Row():
            input_image = gr.Image(type="pil", label="Upload Image", interactive=True)
            output_image = gr.Image(type="pil", label="Segmented Mask")
        
        x_coord = gr.Number(label="X Coordinate")
        y_coord = gr.Number(label="Y Coordinate")
        
        submit = gr.Button("Segment")
        submit.click(segment_image, inputs=[input_image, x_coord, y_coord], outputs=output_image)

    return demo

# Run the demo
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
