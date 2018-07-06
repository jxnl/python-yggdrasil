import torch.nn as nn


class SoftmaxAdapter(nn.Module):
    def __init__(self, dim_in, n_class):
        super(SoftmaxAdapter, self).__init__()
        self.fc = nn.Linear(dim_in, n_class)

    def forward(self, input):
        h = self.fc(input)
        return nn.functional.softmax(h)


class SigmoidAdapter(nn.Module):
    def __init__(self, dim_in, n_class):
        super(SigmoidAdapter, self).__init__()
        self.fc = nn.Linear(dim_in, n_class)

    def forward(self, input):
        h = self.fc(input)
        return nn.functional.sigmoid(h)
