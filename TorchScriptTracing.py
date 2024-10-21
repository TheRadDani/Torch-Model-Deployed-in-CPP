import os
import torch
import torchvision

model_path_save = os.getenv('SERIALIZED_MODEL_FILE', 'traced_resnet_model.pt')

# instance of the model
model = torchvision.models.resnet18()

data = torch.rand(1, 3, 224, 224)

print(model(data)[0][:5])

# use torch.jit.trace to generate a torch.jit.ScriptModule via tracing
traced_script_module = torch.jit.trace(model, data)

traced_script_module.save(model_path_save)
