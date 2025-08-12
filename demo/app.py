# change the path of this file to ../
import sys
sys.path.append("../")
from flask import Flask, request, jsonify
import torch
from segment_anything import SamPredictor, sam_model_registry


app = Flask(__name__)

# Load SAM model
sam = sam_model_registry["vit_h"](checkpoint="../checkpoints/sam_vit_h_4b8939.pth")
sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)

@app.route("/segment", methods=["POST"])
def segment():
    data = request.json
    # Add logic to process input points/boxes and return masks
    return jsonify({"masks": [...]})

if __name__ == "__main__":
    app.run(port=5620)