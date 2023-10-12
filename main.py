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


import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def pre_process_data():
    """
    Pre-processes the data for the GLM model.

    Reads in the data from a CSV file, drops the 'IDpol' column, converts
    categorical variables to category data type, and continuous variables
    to float32 data type. Adds a 'RandN' column with random normal values.
    Splits the data into training and validation sets. Uses a ColumnTransformer
    to scale continuous variables and one-hot encode categorical variables.
    Returns the pre-processed training and validation data.

    Returns:
    X (numpy.ndarray): Pre-processed training data.
    X_val (numpy.ndarray): Pre-processed validation data.
    v (numpy.ndarray): Training exposure data.
    v_val (numpy.ndarray): Validation exposure data.
    y (numpy.ndarray): Training response data.
    y_val (numpy.ndarray): Validation response data.
    """
    data = pd.read_csv("freMTPL2freq.csv")

    response_variable = "ClaimNb"
    exposure_variable = "Exposure"
    categorical_variables = ["Area", "VehBrand", "VehGas", "Region"]
    continuous_variables = ["DrivAge", "Density", "VehPower", "BonusMalus", "VehAge"]

    data = (
        data
        .drop(columns=["IDpol"])
        .assign(
            **{c: lambda x, c=c: x[c].astype("category") for c in categorical_variables},
            **{
                c: lambda x, c=c: x[c].astype("float32")
                for c in [*continuous_variables, response_variable, exposure_variable]
            },
            RandN=lambda x: np.random.normal(size=x.shape[0])
        )
    )

    y = data[[response_variable]].to_numpy()
    v = data[[exposure_variable]].to_numpy()
    X = data.drop(columns=[response_variable, exposure_variable])

    X, X_val, v, v_val, y, y_val = train_test_split(X, v, y)

    transformer = ColumnTransformer(
        [
            ("continuous_pipeline", StandardScaler(), continuous_variables),
            (
                "categorical_pipeline",
                OneHotEncoder(sparse_output=False),
                categorical_variables,
            ),
        ],
        remainder="passthrough",
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )

    X = transformer.fit_transform(X).astype("float32")
    X_val = transformer.transform(X_val).astype("float32")
    return X, X_val, v, v_val, y, y_val


def train_model(X, v, y):
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
        device="cuda",
    )

    X_dict = {"features": X, "exposure": v}
    localglmnet.fit(X_dict, y)
    
    return localglmnet

# writing main function

def main():
    X, X_val, v, v_val, y, y_val = pre_process_data()
    localglmnet = train_model(X, v, y)
    return localglmnet

if __name__ == "__main__":
    main()
    print("completed")
    

