import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, ns, c_bits, c_weight):
        """
        Module for the advanced generator.

        :param ns: multiplier for the number of filters
        :param c_bits: number of bits of a challenge
        :param c_weight: multiplier for the size of the first linear layer
        """
        super().__init__()
        self.c_bits = c_bits
        self.c_weight = c_weight
        self.init_dim = 32
        self.challenge = nn.Linear(self.c_bits, self.init_dim ** 2 * c_weight)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(c_weight, ns * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ns * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ns * 4, ns * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ns * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ns * 2, ns * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ns * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ns * 2, ns, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ns),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ns, 1, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, challenge_input):
        challenge_input = self.challenge(challenge_input)
        challenge_input = challenge_input.view(-1, self.c_weight, self.init_dim, self.init_dim)

        return self.main(challenge_input)
