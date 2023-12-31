import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
DATA_PATH = "./data/freMTPL2freq_clean.csv"

def pre_process_data(verbose=False):
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
    data = pd.read_csv(DATA_PATH)

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

    if verbose:
        print(f"X shape: {X.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"v shape: {v.shape}")
        print(f"v_val shape: {v_val.shape}")
        print(f"y shape: {y.shape}")
        print(f"y_val shape: {y_val.shape}")
    
    return X, X_val, v, v_val, y, y_val


if __name__ == "__main__":
    pre_process_data(verbose=True)
