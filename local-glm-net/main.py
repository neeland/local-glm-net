

import numpy as np
import torch

# Local pacakges
from util import get_device
from data import pre_process_data
from model import train_model

# writing main function
def main():
    """
    Trains a local GLMNET model using pre-processed data and returns the trained model.

    Returns:
    localglmnet: trained GLMNET model
    """
    device = get_device()
    X, X_val, v, v_val, y, y_val = pre_process_data(verbose=True)
    localglmnet = train_model(X, v, y, device)
    ### Extracting attentions &  contributions
    localglmnet_module = localglmnet.module_.to("cpu")
    
    # Get attentions from PyTorch module forward method of by passing attentions = True 
    unscaled_attentions = localglmnet_module(torch.from_numpy(X_val), exposure=v_val, attentions=True).numpy(force=True)
    scaling = localglmnet_module.output_layer.weight.numpy(force=True)
    attentions = unscaled_attentions * scaling
    # Contributions are obtained by simply multiplying the attentions with their corresponding feature value
    contributions = np.multiply(attentions, X_val)
    print("attentions shape: ", attentions.shape)

    ### Extracting the gradients
    return localglmnet

if __name__ == "__main__":
    main()
    print("complete! 🎉")

