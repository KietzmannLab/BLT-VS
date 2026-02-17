
"""
Image preprocessing pipeline for BLT-VS.

This function defines the standard transformation applied to input images
before they are fed into the BLT-VS model.

Processing steps:
-----------------
1. Resize → Resizes the shorter side of the image to 224 pixels.
2. CenterCrop → Crops the image to a 224x224 square.
3. ToTensor → Converts the image to a PyTorch tensor with values in [0, 1].
4. Normalize to [-1, 1] → Scales pixel values from [0, 1] to [-1, 1].

Why this is needed:
-------------------
The BLT-VS model expects fixed-size inputs (224x224) and works best
with inputs normalized around zero. Scaling to [-1, 1] centers the data,
which improves numerical stability and training behavior.

Returns:
--------
transform (torchvision.transforms.Compose):
    A composed transform that can be applied to PIL images before inference.
"""


from torchvision import transforms # type: ignore

def get_blt_vs_transform():
    transform_list = []
    transform_list.append(transforms.Resize(224, antialias=True))
    transform_list.append(transforms.CenterCrop(224))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Lambda(lambda x: 2 * x - 1))
    transform = transforms.Compose(transform_list)
    return transform