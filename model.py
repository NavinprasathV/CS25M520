import math
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}

# Change the architecture here: numbers = conv channels, 'M' = max-pool
VGG6_CFG = [64, 64, "M", 128, 128, "M"]


class VGG6(nn.Module):
    def __init__(self, activation="relu", cfg=VGG6_CFG, num_classes=10):
        super().__init__()
        layers, in_ch = [], 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(in_ch, v, 3, padding=1),
                           nn.BatchNorm2d(v),
                           ACTIVATIONS[activation]()]
                in_ch = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_ch, num_classes)
        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))          # global average pooling
        return self.classifier(x)

    def _init_weights(self):            # same initialisation as the template
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                m.bias.data.zero_()