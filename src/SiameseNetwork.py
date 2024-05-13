import torch
from config import config


class SiameseNetwork(torch.nn.Module):
    def __init__(self, device) -> None:
        super(SiameseNetwork, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Linear(config.embeddingSize, 256).to(self.device)
        self.layer2 = torch.nn.Linear(256, config.comparativeEmbedding).to(self.device)

    def forward(self, x1, x2):
        # a1 = torch.relu(self.layer1(x1))
        # a2 = torch.relu(self.layer1(x2))
        a1 = torch.sigmoid(self.layer1(x1))
        a2 = torch.sigmoid(self.layer1(x2))
        # x = torch.abs(a2 - a1)
        x = a2 - a1
        x = (x - torch.mean(x))/torch.std(x)
        # b = torch.relu(self.layer2(x))
        b = torch.sigmoid(self.layer2(x))
        return b