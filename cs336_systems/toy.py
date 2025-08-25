import torch
from torch import nn
from cs336_basics.nn.utils import cross_entropy_loss


input_shape = (4,32)

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print(f"fc1 dtype {x.dtype} ")
        x = self.ln(x)
        print(f"Layer norm dtype {x.dtype}")
        x = self.fc2(x)
        return x

model = ToyModel(32, 32).cuda()
x = torch.randn(input_shape).cuda()
y = torch.randint(0, input_shape[1], (input_shape[0],)).cuda()

with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    print(f"The parameter dtype is {next(model.parameters()).dtype}")
    logits = model(x)
    print(f"logits dtype {logits.dtype}")
    loss = cross_entropy_loss(logits, y)
    print(f"loss dtype {loss.dtype}")
    loss.backward()
    print(f'The gradient dtype is {next(model.parameters()).grad.dtype}')

    