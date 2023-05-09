from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.functional import jacobian
from torch_examples.model import TwoLayerNN


def differentiable_function(weight_1, bias_1, weight_2, bias_2, model=None):
    pred = model.func(X, weight_1, bias_1, weight_2, bias_2)
    return F.mse_loss(pred, y, reduction="mean")


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


network = TwoLayerNN(2)
mse = nn.MSELoss()

# obtain jacobians using autograd.functional
differentiable_function = partial(differentiable_function, model=network)
jacobians_functional = jacobian(differentiable_function, tuple(network.parameters()))

loss = mse(network(X), y)
for param in network.parameters():
    assert param.grad is None

# obtain jacobians using pytorch
loss.backward()
jacobians_backward = [p.grad.clone().detach() for p in network.parameters()]    # type: ignore[union-attr]

for m, b in zip(jacobians_functional, jacobians_backward):
    assert torch.allclose(m, b)
