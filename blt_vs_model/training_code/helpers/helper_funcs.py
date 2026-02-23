##################
### Importing required packages
##################

import torch
import torchvision.transforms as transforms
import numpy as np
import h5py
import os
from sklearn.linear_model import LinearRegression
import random
from torchvision import datasets, transforms
from collections import Counter

##############################
## Loading the dataset loaders
##############################

def get_Dataset_loaders(hyp, splits):

    # create Datasets for the splits
    if hyp['dataset']['name'] == 'ecoset':

        print('Getting Ecoset ready!')

        dataset_path = hyp['dataset']['dataset_path'] + hyp['dataset']['name'] + '_square256_proper_chunks.h5'
        with h5py.File(dataset_path, "r") as f:
            hyp['dataset']['n_classes'] = np.max(f['val']['labels'][()]) + 1
            hyp['dataset']['class_weights'] = calculate_class_weights_from_h5(f['train']['labels'][()])

        # import the transforms (augmentations)
        transform = get_transform(hyp['dataset']['augment'],hyp)
        transform_val_test = get_transform(hyp['dataset']['augment_val_test'],hyp)

        if 'train' in splits:
            train_data = Ecoset('train', dataset_path=dataset_path, in_memory = 0, transform=transform)
        if 'val' in splits:
            val_data = Ecoset('val', dataset_path=dataset_path, in_memory = 0, transform=transform_val_test)
        if 'test' in splits:
            test_data = Ecoset('test', dataset_path=dataset_path, in_memory = 0, transform=transform_val_test)

    elif hyp['dataset']['name'] == 'imagenet':

        dataset_path = hyp['dataset']['dataset_path'] + 'imagenet'

        print('Getting Imagenet ready!')

        # import the transforms (augmentations)
        transform = get_transform(hyp['dataset']['augment'],hyp)
        transform_val_test = get_transform(hyp['dataset']['augment_val_test'],hyp)      

        if 'train' in splits:
            train_data = datasets.ImageFolder(root=dataset_path + '/train', transform=transform)
            hyp['dataset']['class_weights'] = calculate_class_weights_from_imagefolder(train_data)
        if 'val' in splits:
            val_data = datasets.ImageFolder(root=dataset_path + '/val', transform=transform_val_test) 
        if 'test' in splits:
            test_data = datasets.ImageFolder(root=dataset_path + '/val', transform=transform_val_test) # same as val as we don't have access to the test set

        # Number of classes in ImageNet
        hyp['dataset']['n_classes'] = 1000  # ImageNet has 1000 classes by default

    else:
        print('Dataset not found!')
        return

    print(dataset_path)
    print(f'Number of classes: {hyp["dataset"]["n_classes"]}')

    # create Dataloaders for the splits
    if 'train' in splits:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=hyp['optimizer']['batch_size'], shuffle=True, num_workers=hyp['optimizer']['dataloader']['num_workers_train'], prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor_train'])
    else:
        train_loader = None

    if 'val' in splits:
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=hyp['misc']['batch_size_val_test'], num_workers=hyp['optimizer']['dataloader']['num_workers_val_test'], prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor_val_test'])
    else:
        val_loader = None

    if 'test' in splits:
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=hyp['misc']['batch_size_val_test'],num_workers=hyp['optimizer']['dataloader']['num_workers_val_test'], prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor_val_test'])
    else:
        test_loader = None
        
    return train_loader, val_loader, test_loader, hyp

class Ecoset(torch.utils.data.Dataset):
    #Import Ecoset as a Dataset splitwise

    def __init__(self, split, dataset_path, in_memory=False, transform=None):
        """
        Args:
            dataset_path (string): Path to the .h5 file
            transform (callable, optional): Optional transforms to be applied
                on a sample.
            in_memory: Should we pre-load the dataset?
        """
        self.root_dir = dataset_path
        self.transform = transform
        self.split = split
        self.in_memory = in_memory

        if self.in_memory:
            with h5py.File(dataset_path, "r") as f:
                self.images = torch.from_numpy(f[split]['data'][()]).permute((0, 3, 1, 2)) # to match the CHW expectation of pytorch
                self.labels = torch.from_numpy(f[split]['labels'][()].astype(np.int64))
        else:
            self.split_data = h5py.File(dataset_path, "r")[split]
            self.images = self.split_data['data']
            self.labels = self.split_data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx): # accepts ids and returns the images and labels transformed to the Dataloader
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.in_memory:
            imgs = self.images[idx]
            labels = self.labels[idx]
        else:
            with h5py.File(self.root_dir, "r") as f:
                imgs = torch.from_numpy(np.asarray(self.images[idx])).permute((2,0,1))    
                labels = torch.from_numpy(np.asarray(self.labels[idx].astype(np.int64)))

        if self.transform:
            imgs = self.transform(imgs)

        return imgs, labels
    
def calculate_class_weights_from_h5(labels):
    """
    Calculate class weights for CrossEntropyLoss based on EcoSet labels
    and print the min and max counts per class.

    Args:
        labels (numpy.ndarray): Array of labels for the EcoSet dataset.

    Returns:
        torch.Tensor: Tensor of class weights to use with CrossEntropyLoss.
    """
    # Count occurrences of each class
    class_counts = Counter(labels)

    # Get total number of samples
    total_samples = len(labels)

    # Calculate class weights: inverse proportional to class frequency
    num_classes = len(class_counts)
    class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]

    # Print min and max counts
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    print(f"Minimum count per class: {min_count}")
    print(f"Maximum count per class: {max_count}")

    # Normalize weights (optional)
    class_weights = np.array(class_weights) / sum(class_weights)

    # Convert to a tensor for use in PyTorch
    return torch.tensor(class_weights, dtype=torch.float)

def calculate_class_weights_from_imagefolder(dataset):
    """
    Calculate class weights for CrossEntropyLoss based on the dataset loaded with ImageFolder.

    Args:
        dataset (torchvision.datasets.ImageFolder): Dataset loaded using ImageFolder.

    Returns:
        torch.Tensor: Tensor of class weights to use with CrossEntropyLoss.
    """
    # Get the list of labels for all samples in the dataset
    labels = [sample[1] for sample in dataset.samples]

    # Count occurrences of each class
    class_counts = Counter(labels)

    # Get total number of samples
    total_samples = sum(class_counts.values())

    # Calculate class weights: inverse proportional to class frequency
    num_classes = len(class_counts)
    class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]

    # Print min and max counts
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    print(f"Minimum count per class: {min_count}")
    print(f"Maximum count per class: {max_count}")

    # Normalize weights (optional, depends on preference)
    class_weights = np.array(class_weights) / sum(class_weights)

    # Convert to a tensor for use in PyTorch
    return torch.tensor(class_weights, dtype=torch.float)
    
##############################
## Transform functions
##############################

def get_transform(aug_str,hyp=None):
    # Returns a transform compose function given the transforms listed in "aug_str"

    transform_list = []
    if 'resize_224' in aug_str:
        transform_list.append(transforms.Resize(224, antialias=True))
    if 'crop_224' in aug_str:
        transform_list.append(transforms.RandomCrop(224))
    if 'centercrop_224' in aug_str:
        transform_list.append(transforms.CenterCrop(224))
    if 'resize_128' in aug_str:
        transform_list.append(transforms.Resize(128, antialias=True))
    if 'blurring' in aug_str:
        max_kernel_size = 224//8 - 1
        transform_list.append(RandomGaussianBlur(p=0.5, kernel_size=(1,max_kernel_size), sigma=(0.1,max_kernel_size*1./2))) # apply random gaussian blur "p" of time
    if 'hflip' in aug_str:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if 'trivialaug' in aug_str:
        transform_list.append(transforms.TrivialAugmentWide())
    if 'randaug' in aug_str:
        transform_list.append(transforms.RandAugment())
    if hyp['dataset']['name'] == 'imagenet':
        transform_list.append(transforms.ToTensor())
    else:
        transform_list.append(transforms.ConvertImageDtype(torch.float))
    if 'normalize' in aug_str:
        # transform_list.append(transforms.Lambda(lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1))  # Scale to [-1, 1]
        transform_list.append(transforms.Lambda(lambda x: 2*x - 1))  # to_float, etc. makes images go between [0,1] - the other thing doesn't work as well!

    transform = transforms.Compose(transform_list)
    
    return transform

class RandomGaussianBlur(transforms.GaussianBlur):
    def __init__(self, p, kernel_size, sigma=None):
        super().__init__(kernel_size, sigma)
        self.prob = p

    def __call__(self, img):
        if random.random() < self.prob:  # apply blur if...
            return super().__call__(img)
        return img
    
##############################
## LR scheduler
##############################
    
class LinearFitScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_epochs, factor=1./2, min_lr=1e-8, min_percent_change=1.0, mode='min', patience=5, last_epoch=-1, verbose=False):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            num_epochs (int): Number of epochs to use for the linear fit.
            factor (float): Factor by which the learning rate will be reduced. Default: 0.1.
            min_lr (float): Minimum learning rate. Default: 1e-6.
            min_percent_change (float): Minimum absolute percentage change in the metric to not trigger a reduction. Default: 1.0.
            mode (str): One of 'min' or 'max'. 'min' will reduce the LR if the metric has not decreased by min_percent_change,
                        'max' will reduce the LR if the metric has not increased by min_percent_change. Default: 'min'.
            patience (int): Number of epochs with no improvement after which learning rate will be reduced. Default: 0.
            last_epoch (int): The index of the last epoch. Default: -1.
            verbose (bool): If True, prints a message to stdout for each update. Default: False.
        """
        self.num_epochs = num_epochs
        self.factor = factor
        self.min_lr = min_lr
        self.min_percent_change = min_percent_change
        self.mode = mode
        self.patience = patience
        self.num_bad_epochs = 0  # Track the number of epochs without improvement
        self.verbose = verbose
        self.metric_history = []
        super(LinearFitScheduler, self).__init__(optimizer, last_epoch, verbose)

    def step(self, metric=None):
        """
        Step should be called after each epoch. Can be called without 'metric' during initialization.
        
        Args:
            metric (float, optional): Current epoch's metric. Default is None.
        """
        # Increment the last_epoch attribute from the base class
        self.last_epoch += 1
        
        if metric is not None:
            # Update metric history
            self.metric_history.append(metric)
            
            # Only perform the check if we have enough history
            if len(self.metric_history) >= self.num_epochs:
                # Perform linear fit
                epochs = np.arange(self.num_epochs).reshape(-1, 1)
                metrics = np.array(self.metric_history[-self.num_epochs:]).reshape(-1, 1)
                
                reg = LinearRegression().fit(epochs, metrics)
                slope = reg.coef_[0, 0]
                intercept = reg.intercept_[0]
                
                # Calculate the predicted metrics
                predicted_start = intercept
                predicted_end = slope * (self.num_epochs - 1) + intercept
                
                # Calculate percent change based on the magnitude of the start value
                if predicted_start != 0:
                    percent_change = 100 * (predicted_end - predicted_start) / abs(predicted_start)
                else:
                    percent_change = float('inf')  # Avoid division by zero
                if self.verbose:
                    print(f"Percent_change in metric: {percent_change:.2f}%")
                    
                # Determine if we should adjust the learning rate based on the mode and percent change
                if self.mode == 'min' and percent_change > -self.min_percent_change:
                    self.num_bad_epochs += 1
                elif self.mode == 'max' and percent_change < self.min_percent_change:
                    self.num_bad_epochs += 1
                else:
                    self.num_bad_epochs = 0  # Reset counter if improvement is observed
                
                # Check if we have hit the patience threshold
                if self.num_bad_epochs > self.patience:
                    self.reduce_lr(percent_change)
                    self.metric_history = []  # Reset history after reducing LR
                    self.num_bad_epochs = 0  # Reset bad epoch count after reducing LR

    def reduce_lr(self, percent_change):
        """Reduce the learning rate according to the factor and min_lr constraints and print verbose message."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = max(param_group['lr'] * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            if self.verbose:
                print(f"Reducing learning rate of group {i} to {new_lr:.4e}. Percent change: {percent_change:.2f}%. Patience exceeded.")
    
##############################
## Logging functions
##############################
    
def create_folders_logging(net_name, create_folders=True):

    print('Accessing log folders...')

    log_folder = 'logs/perf_logs'
    net_folder = 'logs/net_params'

    isExist = os.path.exists(log_folder)
    if not isExist and create_folders:
        os.makedirs(log_folder)
        print('Log folder is created!')
    isExist = os.path.exists(net_folder)
    if not isExist and create_folders:
        os.makedirs(net_folder)
        print('Net folder is created!')

    log_folder_name = log_folder+f'/{net_name}'
    net_folder_name = net_folder+f'/{net_name}'

    isExist = os.path.exists(log_folder_name)
    if not isExist and create_folders:
        os.makedirs(log_folder_name)
        print('Specific log folder is created!')
    isExist = os.path.exists(net_folder_name)
    if not isExist and create_folders:
        os.makedirs(net_folder_name)
        print('Specific net folder is created!')

    return log_folder_name, net_folder_name 