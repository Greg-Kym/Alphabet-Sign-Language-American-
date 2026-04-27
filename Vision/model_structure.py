from torch import nn


class Sign_Model(nn.Module):
    def __init__(self, input, hidden_layers, output):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(in_features=input, out_features=hidden_layers),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layers, out_features=hidden_layers),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layers, out_features=hidden_layers),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layers, out_features=output)
        )

    def forward(self, x):
        return self.linear_stack(x)
