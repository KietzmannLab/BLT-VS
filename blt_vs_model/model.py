"""
Model Factory for BLT-VS

This file defines the entry point for constructing the BLT-VS model.
It acts as a configuration and loading interface between the user and
the core architecture defined in blt_vs.py.

Main responsibilities:
----------------------
- Select dataset-specific hyperparameters (e.g., number of classes)
- Configure architectural options (timesteps, skip connections, etc.)
- Instantiate the BLT_VS model with the correct parameters
- Optionally load pretrained weights

In short:
---------
This file does NOT define the neural architecture itself.
Instead, it builds and configures the BLT_VS model based on user input
and prepares it for training or inference.
"""


import torch # type: ignore
from huggingface_hub import hf_hub_download # type: ignore
from .blt_vs import BLT_VS  # Import your BLT_VS class

def blt_vs_model(pretrained=True, training_dataset='imagenet'):
    """
    Returns the BLT_VS model with the weights.
    """
    
    if training_dataset == 'imagenet':
        params = {
            'timesteps': 6,
            'num_classes': 1000,
            'add_feats': 100,
            'lateral_connections': True,
            'topdown_connections': True,
            'skip_connections': True,
            'bio_unroll': False,
            'image_size': 224,
            'hook_type': 'None',
            'readout_type': 'multi'
        }
    elif training_dataset == 'ecoset':
        params = {
            'timesteps': 12,
            'num_classes': 565,
            'add_feats': 100,
            'lateral_connections': True,
            'topdown_connections': True,
            'skip_connections': True,
            'bio_unroll': True,
            'image_size': 224,
            'hook_type': 'None',
            'readout_type': 'multi'
        }

    model = BLT_VS(**params)

    if pretrained:
        # Load weights
        if training_dataset == 'imagenet':
            weight_path = hf_hub_download(repo_id='novelmartis/blt_vs_model', filename='blt_vs_slt_111_biounroll_0_t_6_readout_multi_dataset_imagenet_num_1.pth')
        elif training_dataset == 'ecoset':
            weight_path = hf_hub_download(repo_id='novelmartis/blt_vs_model', filename='blt_vs_slt_111_biounroll_1_t_12_readout_multi_dataset_ecoset_num_1.pth')
        state_dict = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
    return model