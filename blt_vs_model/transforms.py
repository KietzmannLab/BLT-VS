from torchvision import transforms

def get_blt_vs_transform():
    transform_list = []
    transform_list.append(transforms.Resize(224, antialias=True))
    transform_list.append(transforms.CenterCrop(224))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Lambda(lambda x: 2 * x - 1))
    transform = transforms.Compose(transform_list)
    return transform