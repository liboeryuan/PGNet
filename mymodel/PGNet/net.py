# The complete code will be released after the paper is published.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_ch=1, dim=32, deep_supervision=True):
        super().__init__()
        pass

    def forward(self, x):
        
        return
        
if __name__ == "__main__":
    x = torch.rand(4, 1, 256, 256)
    net = Net(in_ch=3, deep_supervision=True)
    out = net(x)
    for i in out:
        print(i.shape)

