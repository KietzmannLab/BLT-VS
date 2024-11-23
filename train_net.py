##################
### Setting up and training a BLT network modelling the ventral stream
# 224px inputs <-> 5deg visual angle
##################

##################
### Collecting some hyperparameters that can be passed through cmd
##################

import argparse
parser = argparse.ArgumentParser(description='Obtaining hyps')

parser.add_argument('--network', type=str, default='blt_vs') # blt_vs / rn50 / others...
parser.add_argument('--timesteps', type=int, default=6) # 6 is the minimum for no bio_unroll, 12 is the minimum for bio_unroll
parser.add_argument('--identifier', type=int, default=1)
parser.add_argument('--lateral_connections', type=int, default=1)
parser.add_argument('--topdown_connections', type=int, default=1)
parser.add_argument('--skip_connections', type=int, default=1)
parser.add_argument('--bio_unroll', type=int, default=0)
parser.add_argument('--readout_type', type=str, default='multi')

parser.add_argument('--dataset', type=str, default='ecoset')
parser.add_argument('--n_epochs', type=int, default=-1) # if -1 then "on-convergence" condition: training finishes when scheduler hits lr of 1e-6
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--batch_size_val_test', type=int, default=256)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--start_from_epoch', type=int, default=0)

parser.add_argument('--grad_clipping', type=int, default=1)

args = parser.parse_args()

##################
### Importing required packages
##################

import torch
import torch.nn as nn
import numpy as np
import time
from helpers.helper_funcs import get_Dataset_loaders, create_folders_logging, LinearFitScheduler
from models.helper_funcs import get_network_model, get_optimizer, eval_network, compute_accuracy, adaptive_gradient_clipping, calculate_flops
import gc

##################
### Listing all hyperparameters
##################

base_lr = args.learning_rate

# as imagenet images are not square, here we first rescale smaller axis to 224 and then crop.
augmenter_train = {'resize_224','crop_224','blurring','hflip','trivialaug','normalize'}
augemnter_val_test = {'resize_224','centercrop_224','normalize'}

hyp = {
    'dataset': {
        'name': args.dataset, # name of the dataset - ecoset/imagenet
        'dataset_path': '/share/klab/datasets/', # Folder where dataset exists (end with '/')
        'augment': augmenter_train, # Mention augmentations to be used here during training - blurring (always first), trivialaug, autoaugment, randaugment, normalize (always last)
        'augment_val_test': augemnter_val_test, # Mention augmentations to be used here during validation/testing
    },
    'network': {
        'name': args.network, # network to be used
        'identifier': str(args.identifier), # identifier in case we run multiple versions of the net
        'timesteps': args.timesteps, # number of timesteps to unroll the RNN
        'lateral_connections': args.lateral_connections, # whether to use lateral connections
        'topdown_connections': args.topdown_connections, # whether to use topdown connections
        'skip_connections': args.skip_connections, # whether to use skip connections
        'bio_unroll': args.bio_unroll, # whether to unroll the network in a biologically plausible manner
        'readout_type': args.readout_type # whether to use a single or multiple readouts
    },
    'optimizer': {
        'type': 'adam', # optimizer to be used
        'lr': {'base_lr': base_lr, # learning rate
               'warmup_epochs': 5 if args.start_from_epoch==0 else 2, # lr starts at base_lr/(lr scale factor) and scales up linearly for these many epochs
               'lr_scale_factor': 100 if args.start_from_epoch==0 else 1.5, # factor by which to scale the learning rate
               },
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs, # number of epochs (full cycle through the dataset)
        'device': 'cuda', # device to train the network on
        'dataloader': {
            'num_workers_train': 10, # number of cpu workers processing the batches 
            'prefetch_factor_train': 4, # number of batches kept in memory by each worker (providing quick access for the gpu)
            'num_workers_val_test': 3, # do not need lots of workers for val/test
            'prefetch_factor_val_test': 4 
        }
    },
    'misc': {
        'use_amp': True, # use automatic mixed precision during training - forward pass .half(), backward full
        'batch_size_val_test': args.batch_size_val_test,
        'save_logs': 1, # after how many epochs should we save a copy of the logs
        'save_net': 1, # after how many epochs should we save a copy of the net - ensure this is a multiple of save_logs
        'start_from_epoch': args.start_from_epoch # at which epoch to start training (data pulled from epoch before that)
    }
}

##################
### Training and evaluation
##################

print('\nAaaand it begins...\n')

def save_filtered_state_dict(state_dict, save_path): # because FLOP computation adds some keys to the state_dict which are not needed for saving
    # Get the model's state_dict
    state_dict = state_dict
    # Filter out keys containing 'total_ops' and 'total_params'
    filtered_state_dict = {k: v for k, v in state_dict.items() if not ('total_ops' in k or 'total_params' in k)}
    # Save the filtered state_dict
    torch.save(filtered_state_dict, save_path)

if __name__ == '__main__':

    # load the dataset loaders to iterate over for training and eval
    train_loader, val_loader, _, hyp = get_Dataset_loaders(hyp,['train','val'])

    # create the network
    net, net_name = get_network_model(hyp)
    net = net.float()

    # creating folders for logging losses/acc and network weights
    log_path, net_path = create_folders_logging(net_name)
    print(f'Log_folders: {log_path} -- {net_path}')

    # Initialize network weights if not starting from scratch
    if hyp['misc']['start_from_epoch'] > 0:
        load_epoch = hyp['misc']['start_from_epoch']
        print(f'Loading epoch: {load_epoch}')
        net_save_path = f'{net_path}/{net_name}_epoch_{load_epoch}.pth'
        state_dict = torch.load(net_save_path)
        net.load_state_dict(state_dict)
        # load_filtered_state_dict(net, net_save_path)

    # Print the number of FLOPs for one pass
    if not args.network == 'blt_vnet':
        print("\nFLOPs for one pass: {}\n".format(calculate_flops(net, torch.randn(1, 3, 224, 224))))
        net.train()
    print(net)

    # Use DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        print("\nLet's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(hyp['optimizer']['device'])

    # criterion and optimizer setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = get_optimizer(hyp,net)
    scaler = torch.cuda.amp.GradScaler(enabled=hyp['misc']['use_amp']) # this is in service of mixed precision training

    # LR scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, threshold=1e-2, verbose=True) # usual scheduler
    lr_scheduler = LinearFitScheduler(optimizer, num_epochs=5, factor=1./2, min_percent_change=1.0, mode='min', verbose=True, patience=2) # 1% change necessary in 5 epochs, for 2 epochs straight, else drop lr by 1/5
    
    # Warm-up scheduler - this already initialises the lr to base_lr/lr_scale_factor - has an internal counter which it uses to do its updates when .step() is called!
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch_h: ((hyp['optimizer']['lr']['base_lr'] - hyp['optimizer']['lr']['base_lr']/hyp['optimizer']['lr']['lr_scale_factor']) / (hyp['optimizer']['lr']['warmup_epochs']-1)) * epoch_h + hyp['optimizer']['lr']['base_lr']/hyp['optimizer']['lr']['lr_scale_factor'])

    # logging losses and accuracies
    if hyp['misc']['start_from_epoch'] == 0:
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        val_accuracies_all = []
    else:
        log_data = np.load(log_path+'/loss_'+net_name+'.npz')
        train_losses = list(log_data['train_loss'][:hyp['misc']['start_from_epoch']])
        train_accuracies = list(log_data['train_accuracies'][:hyp['misc']['start_from_epoch']])
        val_losses = list(log_data['val_loss'][:hyp['misc']['start_from_epoch']])
        val_accuracies = list(log_data['val_accuracies'][:hyp['misc']['start_from_epoch']])

    # saving the randomly initialized network
    if hyp['misc']['start_from_epoch'] == 0:
        if torch.cuda.device_count() > 1: # given how dataparallel works, we need to save the module's state_dict
            save_filtered_state_dict(net.module.state_dict(), f'{net_path}/{net_name}_epoch_{0}.pth')
            # torch.save(net.module.state_dict(), f'{net_path}/{net_name}_epoch_{0}.pth')
        else:
            save_filtered_state_dict(net.state_dict(), f'{net_path}/{net_name}_epoch_{0}.pth')
            # torch.save(net.state_dict(), f'{net_path}/{net_name}_epoch_{0}.pth')

    print('\nTraining begins here!\n')

    epoch = 1
    training_not_finished = 1

    while training_not_finished: # Looping until we reach the desired number of epochs or convergence

        start = time.time()

        torch.cuda.synchronize()
        
        train_loss_running = 0.0
        train_acc_running = 0.0

        epoch_now = epoch+hyp['misc']['start_from_epoch']
        print(f'Epoch: {epoch_now}')
        print('LR now: ',optimizer.param_groups[0]['lr'])

        epoch_running_init_flag = 0

        # Reset memory stats
        for i in range(torch.cuda.device_count()):
            device = f'cuda:{i}'
            torch.cuda.reset_peak_memory_stats(device)
        
        for images,labels in train_loader:

            imgs = images.to(hyp['optimizer']['device'])
            lbls = labels.to(hyp['optimizer']['device'])

            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=hyp['misc']['use_amp']):
                outputs = net(imgs)
                loss = criterion(outputs[0], lbls.long()) 
                if len(outputs) > 1:
                    for t in range(len(outputs)-1):
                        loss = loss + criterion(outputs[t+1], lbls.long())
                loss = loss/len(outputs)
            
            scaler.scale(loss).backward()
            if args.grad_clipping:
                # Clipping gradients
                scaler.unscale_(optimizer)  # Unscale gradients before clipping
                adaptive_gradient_clipping(net, clip_factor=0.1)
            scaler.step(optimizer)
            scaler.update()

            train_loss_running += loss.item()
            # train_loss_running = train_loss_running
            train_acc_running += np.mean(compute_accuracy(outputs,lbls))

            if epoch_running_init_flag == 0:
                epoch_running_init_flag = 1
                print('Epoch is running - 1 batch done!')
            
        train_losses.append(train_loss_running/len(train_loader))
        train_accuracies.append(train_acc_running/len(train_loader))

        max_mem_allocated = 0
        gpu_count = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            device = f'cuda:{i}'
            max_mem_allocated += torch.cuda.max_memory_reserved(device) / (1024**3)
        print(f'Max GPU(s) memory reserved: {max_mem_allocated} Gb; {gpu_count} GPU(s)')
        
        # getting validation loss and acc
        net.eval()
        val_loss_running, val_acc_running = eval_network(val_loader,net,criterion,hyp) # this works similar to the train computations above
        net.train()
        val_acc_running = val_acc_running/len(val_loader)
        val_loss_running = val_loss_running/len(val_loader)/len(outputs)

        val_losses.append(val_loss_running)
        val_accuracies.append(np.mean(val_acc_running))

        print('Epoch time: ', "{:.2f}".format(time.time() - start), ' seconds')
        
        print(f'Train loss: {train_losses[-1]:.2f}; acc: {train_accuracies[-1]:.2f}%')
        print(f'Val loss: {val_losses[-1]:.2f}; acc: {val_accuracies[-1]:.2f}%; acc_t: {val_acc_running}')

        if (epoch) < hyp['optimizer']['lr']['warmup_epochs']: # updating for next epoch's use!
            warmup_scheduler.step()
        else:
            lr_scheduler.step(val_losses[-1])
        
        if (epoch+hyp['misc']['start_from_epoch']) % hyp['misc']['save_logs'] == 0:
            print('Saving metrics!')
            np.savez(log_path+'/loss_'+net_name+'.npz', train_loss=train_losses, val_loss=val_losses, train_accuracies=train_accuracies, val_accuracies=val_accuracies)
        else:
            print('Not saving metrics!')

        if (epoch+hyp['misc']['start_from_epoch']) % hyp['misc']['save_net'] == 0:
            print(f'Saving network!\n')
            epoch_save = epoch+hyp['misc']['start_from_epoch']
            if torch.cuda.device_count() > 1:
                save_filtered_state_dict(net.module.state_dict(), f'{net_path}/{net_name}_epoch_{epoch_save}.pth')
                # torch.save(net.module.state_dict(), f'{net_path}/{net_name}_epoch_{epoch_save}.pth')
            else:
                save_filtered_state_dict(net.state_dict(), f'{net_path}/{net_name}_epoch_{epoch_save}.pth')
                # torch.save(net.state_dict(), f'{net_path}/{net_name}_epoch_{epoch_save}.pth')
        else:
            print('Not saving network!\n')

        epoch += 1
        if hyp['optimizer']['n_epochs'] > 0:
            if epoch > hyp['optimizer']['n_epochs']:
                training_not_finished = 0
                print('\n Done training! - #epochs completed\n')
        elif hyp['optimizer']['n_epochs'] == -1:
            if optimizer.param_groups[0]['lr'] <= 1e-6:
                training_not_finished = 0
                print('\n Done training! - LR reached 1e-6 i.e. converged\n')

    if torch.cuda.device_count() > 1:
        save_filtered_state_dict(net.module.state_dict(), f'{net_path}/{net_name}.pth')
        # torch.save(net.module.state_dict(), f'{net_path}/{net_name}.pth')
    else:
        save_filtered_state_dict(net.state_dict(), f'{net_path}/{net_name}.pth')
        # torch.save(net.state_dict(), f'{net_path}/{net_name}.pth')

    # getting test loss and acc
    _, _, test_loader, hyp = get_Dataset_loaders(hyp,['test'])
    net.eval()
    test_loss_running, test_acc_running = eval_network(test_loader,net,criterion,hyp)
    test_acc = test_acc_running/len(test_loader)
    print(f'Test accuracies over time (%): {test_acc}')
    print('Saving metrics!')
    np.savez(log_path+'/loss_'+net_name+'.npz', train_loss=train_losses, val_loss=val_losses, train_accuracies=train_accuracies, val_accuracies=val_accuracies, test_accuracies=test_acc)