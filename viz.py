### Extracting attentions &  contributions
localglmnet_module = localglmnet.module_.to("cpu")
unscaled_attentions = localglmnet_module(
    torch.from_numpy(X_val), exposure=v_val, attentions=True
).numpy(force=True)
scaling = localglmnet_module.output_layer.weight.numpy(force=True)
attentions = unscaled_attentions * scaling
contributions = np.multiply(attentions, X_val)


### Extracting the gradients

import torch.autograd as autograd

input_tensor = torch.from_numpy(X_val)
input_tensor.requires_grad = True
attentions = localglmnet_module(input_tensor, exposure=v_val, attentions=True)

n, p = input_tensor.shape
gradients = np.empty((p, n, p))
for i in range(p):
    grad_scaling = torch.ones_like(attentions[:, i])
    gradient_i = autograd.grad(
        attentions[:, i], input_tensor, grad_scaling, create_graph=True
    )
    gradients[i, :, :] = gradient_i[0].numpy(force=True) * scaling
