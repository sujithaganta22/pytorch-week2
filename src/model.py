import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, output_size=3):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.fc(x)