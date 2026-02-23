import argparse
import torch
from helpers.helper_funcs import get_Dataset_loaders
from models.helper_funcs import get_network_model


def build_hyp(args):

    hyp = {
        'dataset': {
            'name': 'ecoset',
            'dataset_path': '',
            'augment': set(),
            'augment_val_test': set(),
        },
        'network': {
            'name': args.network,
            'identifier': str(args.identifier),
            'timesteps': args.timesteps,
            'lateral_connections': args.lateral_connections,
            'topdown_connections': args.topdown_connections,
            'skip_connections': args.skip_connections,
            'bio_unroll': args.bio_unroll,
            'readout_type': args.readout_type
        },
        'optimizer': {
            'device': args.device,
            'batch_size': 64,
            'n_epochs': 1,
            'dataloader': {
                'num_workers_train': 0,
                'prefetch_factor_train': 2,
                'num_workers_val_test': 0,
                'prefetch_factor_val_test': 2
            }
        },
        'misc': {
            'batch_size_val_test': 64
        }
    }

    hyp["dataset_mode"] = args.dataset_mode
    return hyp


def main(args):

    print("Running inference on:", args.device)

    hyp = build_hyp(args)

    print("Loading dataset...")
    _, _, test_loader, hyp = get_Dataset_loaders(hyp, ["test"])

    model, net_name = get_network_model(hyp)

    checkpoint_path = f"logs/net_params/{args.net_name}/{args.net_name}_epoch_{args.epoch}.pth"
    print("Loading checkpoint:", checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
    model.to(args.device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            outputs = model(images)
            preds = torch.argmax(outputs[-1], dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"\nTest Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Must match training architecture exactly
    parser.add_argument("--network", type=str, required=True)
    parser.add_argument("--identifier", type=int, default=1)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--lateral_connections", type=int, default=1)
    parser.add_argument("--topdown_connections", type=int, default=1)
    parser.add_argument("--skip_connections", type=int, default=1)
    parser.add_argument("--bio_unroll", type=int, default=0)
    parser.add_argument("--readout_type", type=str, default="multi")

    # Dataset
    parser.add_argument("--dataset_mode", type=int, required=True)

    # Checkpoint
    parser.add_argument("--net_name", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)

    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    main(args)