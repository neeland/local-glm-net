import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

data = pd.read_parquet("freMTPL2freq.parquet")

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
