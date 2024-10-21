import torch
import torchvision
import os
import torch.nn as nn

model_path_save = os.getenv('SERIALIZED_MODEL_FILE', 'traced_resnet_model.pt')

model = torchvision.models.resnet18()
model.eval()

scripted_model = torch.jit.script(model)

scripted_model.save(model_path_save)
