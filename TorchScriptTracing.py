import torch
import torchvision

# instance of the model
model = torchvision.models.resnet18()

data = torch.rand(1, 3, 224, 224)

# use torch.jit.trace to generate a torch.jit.ScriptModule via tracing
traced_script_module = torch.jit.trace(model, data)

traced_script_module.save("traced_resnet_model.pt")
