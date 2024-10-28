import onnx
import onnxruntime
import torch
import numpy as np

from process_gray_scale_img import *
from load_onnx_model import *

# Run super resolution model in ONNX Runtime

ort_inputs = {ort_session.get_inputs()[0].name : to_numpy(img_y)}
ort_outputs = ort_session.run(None, ort_inputs)
img_out_y = ort_outputs[0]

# Reconstruct the image from output tensor
img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]),
                             mode='L')

# PyTorch post-processing to get the output image
final_img = Image.merge(
    "YCbCr", [
    img_out_y,
    img_cb.resize(img_out_y.size, Image.BICUBIC),
    img_cr.resize(img_out_y.size, Image.BICUBIC),
]).convert("RGB")

final_img.save("./images/cat_superres.jpg")

# Save resized original image
transforms_resize = transforms.Resize([img_out_y.size[0], 
                         img_out_y.size[1]])
img = transforms_resize(img)
img.save("./images/cat_resized.jpg")