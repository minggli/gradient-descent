# mypy: disable-error-code=union-attr
import torch
from torch import nn
from torch.optim import SGD
from torch_examples.model import TwoLayerNN

X = torch.Tensor(
    [
        [1, 2],
        [2, 5],
        [4, 9]
    ]
)
y = torch.Tensor(
    [100, 200, 370]
).unsqueeze(-1)

model = TwoLayerNN(2)
lr = 1e-4
mse = nn.MSELoss()
optim = SGD(model.parameters(), lr)

for epoch in range(20):
    loss = mse(model(X), y)
    optim.zero_grad()

    # gradients are not computed until backward propagation
    for param in model.parameters():
        assert param.grad is None
    loss.backward()
    for param in model.parameters():
        assert param.grad.size() == param.size()

    params_t = [param.clone().detach() for param in model.parameters()]
    jacobians_t = [param.grad.clone().detach() for param in model.parameters()]

    optim.step()
    params_t_1 = [param.clone().detach() for param in model.parameters()]
    # verify negative gradient being applied in direction of fastest descent
    for param_t, jacobian_t, param_t_1 in zip(params_t, jacobians_t, params_t_1):
        assert torch.allclose(param_t_1, param_t - lr * jacobian_t)
