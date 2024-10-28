import onnx
import onnxruntime
import torch
import time
from onnx_runtime import *

onnx_model = onnx.load("super_resolution.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("super_resolution.onnx")
providers = ["CPUExecutionProvider"]

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else \
        tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
out_outputs = ort_session.run(None, ort_inputs)

# Compare ONNX runtime output predction
np.testing.assert_allclose(to_numpy(torch_out), out_outputs[0], \
                           rtol=1e-03, atol=1e-05)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")

# meassure runtime time

start = time.time()

torch_out = torch_model(x)
end = time.time()
print(f"Inference of Pytorch model used {end - start} seconds")

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
start = time.time()
ort_outs = ort_session.run(None, ort_inputs)
end = time.time()
print(f"Inference of ONNX model used {end - start} seconds")