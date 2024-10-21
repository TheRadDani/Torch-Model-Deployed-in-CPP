import torch
import torch.nn as nn

class VanillaModule(nn.Module):
    def __init__(self, n, m):
        super(VanillaModule, self).__init__()
        self.weight = nn.Parameter(torch.rand(n, m))

    def forward(self, input):
        if input.sum() > 0:
            output = self.weight.mv(input)
        else:
            output = self.weight + input
        
        return output

module = VanillaModule(10, 20)
sm = torch.jit.script(module)
