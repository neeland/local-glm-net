import torch
def get_device():
    """
    Returns the device to use for training.

    Returns:
    torch.device: Device to use for training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"torch device: {device}")
    return device
