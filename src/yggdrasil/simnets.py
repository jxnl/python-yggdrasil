import torch.nn as nn


class TripletNetwork(nn.Module):
    def __init__(self, model):
        super(TripletNetwork, self).__init__()
        self.model = model

    def forward_single(self, input):
        return self.model(input)

    def forward(self, *inputs):
        assert len(inputs) == 3
        return [self.model(x) for x in inputs]


class SiameseNetwork(nn.Module):
    def __init__(self, model):
        super(SiameseNetwork, self).__init__()
        self.model = model

    def forward_single(self, input):
        return self.model(input)

    def forward(self, *inputs):
        assert len(inputs) == 2
        return [self.model(x) for x in inputs]
