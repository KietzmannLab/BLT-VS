dependencies = ['torch', 'torchvision', 'numpy', 'huggingface_hub']

def blt_vs_model(pretrained=True, training_dataset='imagenet'):
    """
    Returns the BLT_VS model with optional pre-trained weights.
    """
    from blt_vs_model.model import blt_vs_model
    return blt_vs_model(pretrained=pretrained, training_dataset='imagenet')

def get_blt_vs_transform():
    """
    Returns the required image transforms for the BLT_VS model.
    """
    from blt_vs_model.transforms import get_blt_vs_transform
    return get_blt_vs_transform()