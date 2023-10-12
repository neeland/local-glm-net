from skorch import NeuralNetRegressor

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
