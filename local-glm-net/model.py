

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


import torch
from torch import nn
from torch.optim import NAdam
from skorch import NeuralNetRegressor


class LocalGLMNet(nn.Module):
    """
    A neural network model for generalized linear models with local linear
    approximation. The model consists of a series of fully connected hidden
    layers with a tanh activation function, followed by a skip connection and
    an output layer with an exponential activation function. The skip connection
    is computed as the dot product between the output of the last hidden layer
    and the input features. The output of the model is the element-wise product
    of the output of the output layer and an exposure parameter (if provided).
    """
    
    def __init__(self, input_size, hidden_layer_sizes):
        super(LocalGLMNet, self).__init__()

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(input_size, hidden_layer_sizes[0])]
        )
        self.hidden_layers.extend(
            [
                nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
                for i in range(len(hidden_layer_sizes) - 1)
            ]
        )
        self.last_hidden_layer = nn.Linear(hidden_layer_sizes[-1], input_size)
        self.output_layer = nn.Linear(1, 1)
        self.activation = nn.Tanh()
        self.inverse_link = torch.exp

    def forward(self, features, exposure=None, attentions=False):
            """
            Forward pass of the model.

            Args:
                features (torch.Tensor): Input features.
                exposure (torch.Tensor, optional): Exposure variable. Defaults to None.
                attentions (bool, optional): Whether to return attention weights. Defaults to False.

            Returns:
                torch.Tensor: Model output.
            """
            x = features
            for layer in self.hidden_layers:
                x = self.activation(layer(x))
            x = self.last_hidden_layer(x)
            if attentions:
                return x
            # Dot product
            skip_connection = torch.einsum("ij,ij->i", x, features).unsqueeze(1)
            x = self.output_layer(skip_connection)
            x = self.inverse_link(x)
            if exposure is None:
                exposure = torch.ones_like(x, device=features.device)
            x = x * exposure
            return x

def train_model(X, v, y, device):
    """
    Trains a LocalGLMNet model using the provided input data and returns the trained model.

    Args:
        X (numpy.ndarray): Input features of shape (n_samples, n_features).
        v (numpy.ndarray): Exposure variable of shape (n_samples,).
        y (numpy.ndarray): Target variable of shape (n_samples,).
        device: Device to use for training.

    Returns:
        NeuralNetRegressor: Trained LocalGLMNet model.
    """
    localglmnet = NeuralNetRegressor(
        module=LocalGLMNet,
        max_epochs=10,
        criterion=nn.PoissonNLLLoss,
        criterion__log_input=False,
        module__input_size=X.shape[1],
        module__hidden_layer_sizes=[64, 32, 16],
        optimizer=NAdam,
        lr=0.01,
        batch_size=512,
        device=device,
    )

    X_dict = {"features": X, "exposure": v}
    localglmnet.fit(X_dict, y)
    return localglmnet