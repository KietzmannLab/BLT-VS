import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

##############################################################################################
# B modelling ventral stream
##############################################################################################

class vNet(nn.Module):

    def __init__(self, num_classes=565):

        # for info on this model, see https://www.pnas.org/doi/10.1073/pnas.2011417118#sec-3 ; also check - https://codeocean.com/capsule/9570390/tree/v1

        super(vNet, self).__init__()

        self.num_classes = num_classes

        # Network area definitions with bottom-up pass info
        self.kernel_sizes = [7, 7, 5, 5, 3, 3, 3, 3, 1, 1]
        self.channel_sizes = [128, 128, 256, 256, 512, 512, 1024, 1024, 2048, 2048]
        self.maxpool = [False, False, True, True, False, False, True, True, True, True]

        # Collecting connections
        self.connections = nn.ModuleDict()
        for idx in range(len(self.kernel_sizes)):
            self.connections[f'{idx}'] = nn.Conv2d(
                in_channels=self.channel_sizes[idx-1],
                out_channels=self.channel_sizes[idx],
                kernel_size=self.kernel_sizes[idx],
                padding='same',
            ) if idx != 0 else nn.Conv2d(
                in_channels=3, # assuming input is RGB
                out_channels=self.channel_sizes[idx],
                kernel_size=self.kernel_sizes[idx],
                padding='same',
            )

        # Readout layer
        self.connections['Readout'] =  nn.Linear(self.channel_sizes[-1], self.num_classes)

        # Collecting layernorms
        self.layernorms = nn.ModuleDict()
        for idx in range(len(self.kernel_sizes)):
            self.layernorms[f'norm_{idx}'] = nn.GroupNorm(num_groups=1, num_channels=self.channel_sizes[idx])
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        # Bottom-up pass
        for idx in range(len(self.kernel_sizes)):
            if self.maxpool[idx]:
                x = F.max_pool2d(x, 2)
            # print(x.shape)
            x = self.connections[f'{idx}'](x)
            x = F.relu(x)
            x = self.layernorms[f'norm_{idx}'](x)
        x = self.global_avg_pool(x)
        x = torch.squeeze(x)
        x = self.connections['Readout'](x)

        return [x]
    
