from torch import nn

class SimpleModel(nn.Module):
    def __init__(self, num_features):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        return x
