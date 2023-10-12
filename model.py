import torch
from torch import nn
from torch.optim import NAdam


class LocalGLMNet(nn.Module):
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
