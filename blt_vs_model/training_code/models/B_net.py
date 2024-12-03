import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

##############################################################################################
# B modelling ventral stream
##############################################################################################

class B_VS(nn.Module):

    def __init__(self, num_classes=565, image_size=224):

        super(B_VS, self).__init__()

        # Retina -> LGN -> V1 -> V2 -> V3 -> V4 -> IT -> Readout: kernel size and strides computed to match mean RF sizes for each ROI, assuming a 5deg image presentation for 224px images (additionally there's a provision for 128px in case you need a smaller network), channel size computed using #neurons comparison
        # Avg. RF sizes (radii at 2.5deg): Retina: 0.18deg (Table 1, P-cells-surround https://www.sciencedirect.com/science/article/pii/0042698994E0066T), LGN: 0.4 (Fig.5 https://www.jneurosci.org/content/22/1/338.full), V1: 0.75, V2: 0.8, V3: 1.37, V4: 1.85, LOC: 2.48 (Fig. 4 https://www.jneurosci.org/content/jneuro/31/38/13604.full.pdf)
        # Avg. #neurons: Retina: 300k, LGN: 450k, V1: 98M, V2: 68.6M, V3: 39.2M, V4: 19.6M, LOC: 39.2M (see txt in folder - extremely crude approximations using chatgpt-o1-preview). Now there's 4 kinds of pyramidal cells in cortical columns it seems (see Fig in Box 1 https://www.nature.com/articles/nature12654) so we divide the numbers by half as we only care about the bottom-up and top-down pathways. So scaling is [1,1,326,229,131,65,131] - too large so I'll use relative scale - based roughly on square root - [1,1,18,15,11,8,11]

        # Eff RFs and area sizes (224px):
        # layer 0 - retina - RF size needed 8 - kernel size 7 - got RF 7 - stride 2 (eff stride 2, 112px)
        # layer 1 - LGN - RF size needed 18 - kernel size 7 - got RF 19 - stride 2 (eff stride 4, 56px)
        # layer 2 - V1 - RF size needed 34 - kernel size 5 - got RF 35 - stride 2 (eff stride 8, 28px)
        # layer 3 - V2  - RF size 36 - kernel size 1 - got RF 35 (eff stride 8, 28px)
        # layer 4 - V3 - RF size 61 - kernel size 5 - got RF 67 (eff stride 8, 28px)
        # layer 5 - V4 - RF size 83 - kernel size 3 - got RF 83 (eff stride 8, 28px)
        # layer 6 - LOC - RF size 93 - kernel size 3 - got RF 99 - stride 2 (eff stride 16, 14px)
        # layer 7 - Readout - kernel size 5 - got RF 163 - stride 2 (eff stride 32, 7px) - 163/224 is good enough for good object-scale readout!

        # Eff RFs and area sizes (128px):
        # layer 0 - retina - RF size needed 5 - kernel size 5 - got RF 5 - stride 2 (eff stride 2, 64px)
        # layer 1 - LGN - RF size needed 10 - kernel size 3 - got RF 9 - stride 2 (eff stride 4, 32px)
        # layer 2 - V1 - RF size needed 19 - kernel size 3 - got RF 17 - stride 2 (eff stride 8, 16px)
        # layer 3 - V2  - RF size 20 - kernel size 1 - got RF 17 (eff stride 8, 16px)
        # layer 4 - V3 - RF size 35 - kernel size 3 - got RF 33 (eff stride 8, 16px)
        # layer 5 - V4 - RF size 47 - kernel size 3 - got RF 49 (eff stride 8, 16px)
        # layer 6 - LOC - RF size 63 - kernel size 3 - got RF 65 - stride 2 (eff stride 16, 8px)
        # layer 7 - Readout - kernel size 3 - got RF 97 - stride 2 (eff stride 32, 4px) - 97/128 is good enough for good object-scale readout!

        self.num_classes = num_classes
        self.image_size = image_size

        # Network area definitions with bottom-up pass info
        self.areas = ['Retina', 'LGN', 'V1', 'V2', 'V3', 'V4', 'LOC','Readout']
        kernel_sizes = [7, 7, 5, 1, 5, 3, 3, 5] if image_size==224 else [5, 3, 3, 1, 3, 3, 3, 3]
        strides = [2, 2, 2, 1, 1, 1, 2, 2]
        paddings = (np.array(kernel_sizes)-1)//2 # to maintain some semblance of padding='same'
        channel_sizes = [32,32,576,480,352,256,352,int(num_classes)]

        # Collecting connections
        self.connections = nn.ModuleDict()
        for idx in range(len(self.areas) - 1):
            area = self.areas[idx]
            self.connections[f'{area}'] = nn.Conv2d(
                in_channels=3, # assuming input is RGB
                out_channels=channel_sizes[idx],
                kernel_size=kernel_sizes[idx],
                stride=strides[idx],
                padding=paddings[idx]
            ) if idx == 0 else nn.Conv2d(
                in_channels=channel_sizes[idx-1],
                out_channels=channel_sizes[idx],
                kernel_size=kernel_sizes[idx],
                stride=strides[idx],
                padding=paddings[idx]
            )
        # Readout layer
        layer_n = 7
        self.connections['Readout'] =  nn.Conv2d(
            in_channels=channel_sizes[layer_n-1],
            out_channels=channel_sizes[layer_n],
            kernel_size=kernel_sizes[layer_n],
            stride=strides[layer_n],
            padding=(kernel_sizes[layer_n] - 1) // 2 
        )

        # Collecting layernorms
        self.layernorms = nn.ModuleDict()
        for idx in range(len(self.areas)):
            self.layernorms[f'norm_{self.areas[idx]}'] = nn.GroupNorm(num_groups=1, num_channels=channel_sizes[idx])
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        # Bottom-up pass
        for area in self.areas:
            x = self.connections[area](x)
            x = F.relu(x)
            x = self.layernorms[f'norm_{area}'](x)
        x = self.global_avg_pool(x)
        x = torch.squeeze(x)

        return [x]
    
    def get_activation(self, x, layer):

        for area in self.areas:
            x = self.connections[area](x)
            x = F.relu(x)
            x = self.layernorms[f'norm_{area}'](x)
            if area == layer:
                return x
    
