import torch

from torch import Tensor
from torchmetrics import Metric


class MedianValue(Metric):
    def __init__(self):
        super(MedianValue, self).__init__()

        self.add_state("values", default=[])

    def update(self, values: Tensor):
        self.values.append(values)

    def compute(self):
        values = self.values[0]
        for i in range(1, len(self.values)):
            values = torch.cat((values, self.values[i]))

        return torch.median(values)


def main():
    metric = MedianValue()

    for i in range(11):
        value = torch.randn(torch.randint(low=1, high=100, size=(1,)))
        metric(value)

    print(metric.compute())


if __name__ == "__main__":
    main()
