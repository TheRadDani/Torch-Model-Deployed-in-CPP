import numpy as np

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
from super_resolution_model import SuperResolutionNet


# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the super-resolution model using the model definition
torch_model = SuperResolutionNet(upscale_factor=3).to(device)

# Load pretrained model weights
model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'

# Initialize model with the pretrained weights

map_location = None if torch.cuda.is_available() else lambda storage, loc: storage

torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

# set the model to inference mode
torch_model.eval()

# export the model via tracing in order to export 
# the model in Onnx

batch_size = 1    # just a random number
# Input to the model
x = torch.randn(batch_size, 1, 224, 224, requires_grad=True).to(device)

torch_out = torch_model(x)

# Export the model
torch.onnx.export(
    torch_model,               # Model being run
    x,                         # Model input (or a tuple for multiple inputs)
    "super_resolution.onnx",   # Path to save the model (can be a file or file-like object)
    export_params=True,        # Store the trained parameter weights inside the model file
    opset_version=11,          # ONNX version to export the model to
    do_constant_folding=True,  # Optimize the model by constant folding
    input_names=['input'],     # Model's input names
    output_names=['output'],   # Model's output names
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},  # Variable length axes
)

print("Model exported to ONNX format successfully.")

