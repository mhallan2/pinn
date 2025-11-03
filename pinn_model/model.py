import torch.nn as nn


class PINN(nn.Module):
    """
    Реализация Physics-Informed Neural Networks (PINNs) на PyTorch.

    PINN — это подход к решению дифференциальных уравнений, который:

    1. Аппроксимирует неизвестное решение u(x) нейронной сетью
    2. Учитывает дифференциальное уравнение в виде слагаемого в функции потерь
    3. Использует автоматическое дифференцирование для вычисления производных
    4. Не требует данных о решении, только уравнение и граничные условия
    """

    def __init__(
        self,
        input_size=2,
        hidden_sizes=(32, 64, 32),
        output_size=1,
        activation=nn.Tanh(),
    ):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_sizes[0]), activation]
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)
