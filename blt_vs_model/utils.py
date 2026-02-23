import json
import sys

if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    import importlib_resources as importlib_resources

def load_class_names(dataset='imagenet'):
    """
    Load class names from the specified dataset JSON file.
    Args:
        dataset (str): Name of the dataset ('imagenet' or 'ecoset').
    Returns:
        class_names (list): List of class names indexed by class IDs.
    """
    valid_datasets = ['imagenet', 'ecoset']
    if dataset not in valid_datasets:
        raise ValueError(f"Dataset must be one of {valid_datasets}")

    filename = f'{dataset}.json'

    if sys.version_info >= (3, 9):
        data_path = files('blt_vs_model') / filename
        with data_path.open('r', encoding='utf-8') as f:
            class_names = json.load(f)
    else:
        with importlib_resources.open_text('blt_vs_model', filename, encoding='utf-8') as f:
            class_names = json.load(f)

    return class_names