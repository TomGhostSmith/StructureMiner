import torch
from config import config

class Classifier(torch.nn.Module):
    def __init__(self, algorithm, device) -> None:
        super(Classifier, self).__init__()
        self.algorithm = algorithm
        self.device = device
        if (algorithm == 'LR'):
            self.linear = torch.nn.Linear(config.comparativeEmbedding, 1).to(self.device)
        elif (algorithm == 'SVM'):
            self.linear = torch.nn.Linear(config.comparativeEmbedding, 1).to(self.device)
            self.margin = 1

    def forward(self, x):
        if (self.algorithm == 'LR'):
            result = torch.sigmoid(self.linear(x))
        elif (self.algorithm == 'SVM'):
            result = self.linear(x)

        self.output = result
        return result

    def getLoss(self, trueValue):
        if (self.algorithm == 'LR'):
            criterion = torch.nn.BCELoss()
            loss = criterion(self.output, trueValue)
        elif (self.algorithm == 'SVM'):
            trueValue = 2 * trueValue - 1  # convert 0/1 to -1/+1
            loss = torch.mean(torch.max(0, self.margin - trueValue * self.output))
        return loss