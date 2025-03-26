import pickle

import torch


def load_network_pkl(f, device='cpu', verbose=True):
    """
    Load a StyleGAN2-ADA network snapshot (.pkl) file.

    Args:
        f (str or file-like): Path to the .pkl file or file object.
        device (str): 'cpu' or 'cuda' or torch.device
        verbose (bool): If True, print model info (z_dim, c_dim, structure)

    Returns:
        dict: Dictionary with 'G', 'D', 'G_ema' keys (if present)
    """
    if isinstance(f, str):
        with open(f, 'rb') as file:
            data = pickle.load(file)
    else:
        data = pickle.load(f)

    # Validate presence of expected keys
    if not isinstance(data, dict):
        raise RuntimeError('Invalid .pkl file format: expected dict.')

    # found_keys = list(data.keys())
    if not any(k in data for k in ['G', 'G_ema']):
        raise RuntimeError("No 'G' or 'G_ema' found in the .pkl file.")

    models = {}
    for key in ['G', 'D', 'G_ema']:
        if key in data and isinstance(data[key], torch.nn.Module):
            models[key] = data[key].to(device)

    # Print model info if verbose
    if 'G_ema' in models and verbose:
        G = models['G_ema']
        z_dim = getattr(G, 'z_dim', 'Unknown')
        c_dim = getattr(G, 'c_dim', 'Unknown')
        print(f"[INFO] Loaded G_ema with z_dim={z_dim}, c_dim={c_dim}")
        print(f"[INFO] Model architecture:{G}")

    return models
