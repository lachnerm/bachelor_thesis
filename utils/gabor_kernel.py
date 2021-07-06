# kernel that is used for the second Gabor transformation
gabor_kernel = [[-0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000,
                 -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, 0.500000,
                 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000,
                 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, -0.250000, -0.250000, -0.250000, -0.250000,
                 -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000,
                 -0.250000, -0.250000, -0.250000, -0.250000],
                [-0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000,
                 -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, 0.500000,
                 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000,
                 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, -0.250000, -0.250000, -0.250000, -0.250000,
                 -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000,
                 -0.250000, -0.250000, -0.250000, -0.250000],
                [-0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000,
                 -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, 0.500000,
                 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000,
                 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, -0.250000, -0.250000, -0.250000, -0.250000,
                 -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000,
                 -0.250000, -0.250000, -0.250000, -0.250000],
                [-0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000,
                 -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, 1.000000,
                 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, -0.500000, -0.500000, -0.500000, -0.500000,
                 -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000,
                 -0.500000, -0.500000, -0.500000, -0.500000],
                [-0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000,
                 -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, 1.000000,
                 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, -0.500000, -0.500000, -0.500000, -0.500000,
                 -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000,
                 -0.500000, -0.500000, -0.500000, -0.500000],
                [-0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000,
                 -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, 1.000000,
                 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, -0.500000, -0.500000, -0.500000, -0.500000,
                 -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000, -0.500000,
                 -0.500000, -0.500000, -0.500000, -0.500000],
                [-0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000,
                 -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, 0.500000,
                 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000,
                 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, -0.250000, -0.250000, -0.250000, -0.250000,
                 -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000,
                 -0.250000, -0.250000, -0.250000, -0.250000],
                [-0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000,
                 -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, 0.500000,
                 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000,
                 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, -0.250000, -0.250000, -0.250000, -0.250000,
                 -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000,
                 -0.250000, -0.250000, -0.250000, -0.250000],
                [-0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000,
                 -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, 0.500000,
                 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000,
                 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, -0.250000, -0.250000, -0.250000, -0.250000,
                 -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000, -0.250000,
                 -0.250000, -0.250000, -0.250000, -0.250000]]