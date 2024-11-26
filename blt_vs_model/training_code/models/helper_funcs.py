import torch
import torch.optim as optim
import numpy as np
from thop import profile
    
##################################
## Importing the network
##################################

def get_network_model(hyp):
    # import the req. network

    timesteps = hyp['network']['timesteps']
    netnum = hyp['network']['identifier']
    num_classes = int(hyp['dataset']['n_classes'])
    lateral_connections = hyp['network']['lateral_connections']
    topdown_connections = hyp['network']['topdown_connections']
    skip_connections = hyp['network']['skip_connections']
    bio_unroll = hyp['network']['bio_unroll']
    network = hyp['network']['name']
    dataset = hyp['dataset']['name']
    readout_type = hyp['network']['readout_type']

    if network == 'blt_vs':

        from .BLT_net import BLT_VS
        net = BLT_VS(timesteps = timesteps, num_classes=num_classes, lateral_connections=lateral_connections, topdown_connections=topdown_connections,skip_connections=skip_connections, bio_unroll=bio_unroll, readout_type=readout_type)
        net_name = f'{network}_slt_{skip_connections}{lateral_connections}{topdown_connections}_biounroll_{bio_unroll}_t_{timesteps}_readout_{readout_type}_dataset_{dataset}_num_{netnum}'

    if network == 'b_vs':

        from .B_net import B_VS
        net = B_VS(num_classes=num_classes)
        net_name = f'{network}_dataset_{dataset}_num_{netnum}'
    
    elif network == 'rn50':

        from .ResNet import ResNet50
        net = ResNet50(num_classes=num_classes)
        net_name = f'{network}_dataset_{dataset}_num_{netnum}'

    elif network == 'blt_vnet':

        from .klab_models import BLT_Kietzmannlab
        net = BLT_Kietzmannlab(n_classes=num_classes,
        n_recurrent_steps=timesteps,
        double_decker=True,
        divide_n_channels=1,
        norm_type="LN",
        l_flag=lateral_connections,
        t_flag=topdown_connections,
        lt_interact=1, # 0 is additive, 1 is multiplicative
        LT_position="all",
        classifier_bias=True,
        relu_last = True)
        net_name = f'{network}_dataset_{dataset}_num_{netnum}'

    elif network == 'cornet_s':

        from .CORnet import CORnet_S
        net = CORnet_S(num_classes=num_classes)
        net_name = f'{network}_dataset_{dataset}_num_{netnum}'

    print(f'\nNetwork name: {net_name}')     
    
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"\nThe network has {params} trainable parameters\n")

    return net, net_name

##################################
## Other functions
##################################

def calculate_flops(model, input_tensor, custom_ops=None):

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        flops, _ = profile(model, inputs=(input_tensor,), custom_ops=custom_ops, verbose=False)
    # Format the FLOPs value in scientific notation with 3 decimal places
    flops_formatted = "{:.3e}".format(flops)

    return flops_formatted

def get_optimizer(hyp,net):
    # selecting the optimizer

    # Get the parameters that will be updated (those with requires_grad=True) - this is set up so later you could freeze some layers if needed
    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))

    if hyp['optimizer']['type'] == 'adam':
        return optim.Adam(trainable_params,lr=1.)

def adaptive_gradient_clipping(model, clip_factor=0.1, eps=1e-3):
    # Adaptive gradient clipping from https://proceedings.mlr.press/v139/brock21a.html
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.data.norm(2)
            grad_norm = param.grad.data.norm(2)
            max_norm = clip_factor * (param_norm + eps)
            if grad_norm > max_norm:
                scaling_factor = max_norm / (grad_norm + eps)
                param.grad.data.mul_(scaling_factor)
    
def compute_accuracy(outputs,labels):
    
    timesteps = len(outputs)
    accuracies = [0 for t in range(timesteps)]
    
    for t in range(timesteps):
        _, predicted = torch.max(outputs[t].data, 1)
        total = labels.shape[0]
        correct = (predicted == labels).sum().item()
        accuracies[t] = correct*100./total
    
    return accuracies 
    
def eval_network(data_loader,net,criterion,hyp): 
    # during training, evaluate network on val or test images (val_loader or test_loader)

    with torch.no_grad():
        loss_running = 0.0
        acc_running = []
        for images,labels in data_loader:
            imgs = images.to(hyp['optimizer']['device'])
            lbls = labels.to(hyp['optimizer']['device'])
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=hyp['misc']['use_amp']):
                outputs = net(imgs)
                loss = criterion(outputs[0], lbls.long())
                if len(acc_running) == 0:
                    acc_running = np.zeros([len(outputs),])
                acc_running[0] += compute_accuracy([outputs[0]],lbls)[0]
                if len(outputs) > 1:
                    for t in range(len(outputs)-1):
                        loss = loss + criterion(outputs[t+1], lbls.long())
                        acc_running[t+1] += compute_accuracy([outputs[t+1]],lbls)[0]
            loss_running += loss.item()
            loss_running = loss_running

    return loss_running, acc_running
