import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, c_bits, ns):
        """
        Module for the generator.

        :param c_bits: number of bits of a challenge
        :param ns: multiplier for the number of filters
        """
        super().__init__()
        self.c_bits = c_bits

        slope = 0.2
        self.main = nn.Sequential(
            nn.ConvTranspose2d(c_bits, ns * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ns * 32),
            nn.LeakyReLU(slope, inplace=True),

            nn.ConvTranspose2d(ns * 32, ns * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ns * 16),
            nn.LeakyReLU(slope, inplace=True),

            nn.ConvTranspose2d(ns * 16, ns * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ns * 8),
            nn.LeakyReLU(slope, inplace=True),

            nn.ConvTranspose2d(ns * 8, ns * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ns * 4),
            nn.LeakyReLU(slope, inplace=True),

            nn.ConvTranspose2d(ns * 4, ns * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ns * 2),
            nn.LeakyReLU(slope, inplace=True),

            nn.ConvTranspose2d(ns * 2, ns, 8, 4, 2, bias=False),
            nn.BatchNorm2d(ns),
            nn.LeakyReLU(slope, inplace=True),

            nn.ConvTranspose2d(ns, 1, 8, 4, 2, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        input = input.view(-1, self.c_bits, 1, 1)
        return self.main(input)
