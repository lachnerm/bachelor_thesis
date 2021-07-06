import itertools

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_ssim import ssim

from non_linear.modules.cropped.Cropped_Generator_type1_module import Generator as Generator_Cropped_type1
from non_linear.modules.cropped.Cropped_Generator_type2_module import Generator as Generator_Cropped_type2
from non_linear.modules.normal.Generator_module import Generator
from non_linear.utils.utils import calc_pc, weights_init, calc_ssim
from utils import utils


class GeneratorModel(LightningModule):
    """
    DL model for the generator attack. After training, the pearson correlation coefficient, structural similarity index
    and fractional hamming distance for each real and predicted response pair of the test set are computed.
    """

    def __init__(self, hparams, img_size, crop_size, c_bits, denormalize, do_crop, crop_type2, custom_gabor):
        """
        Initializes the generator.

        :param hparams: hyperparameters that are used for the model
        :param img_size: size of the responses
        :param crop_size: size to which the responses will be cropped if decided to
        :param c_bits: number of bits of a challenge
        :param denormalize: inverse function of the normalization applied to the responses for the DL attack
        :param do_crop: whether to crop the responses
        :param crop_type2: whether to use the cropped generator type 2
        :param custom_gabor: whether to use the second gabor transformation
        """
        super().__init__()
        self.hparams = hparams
        self.c_bits = c_bits
        self.denormalize = denormalize
        self.crop = do_crop
        self.img_size = img_size
        self.crop_size = crop_size
        self.custom_gabor = custom_gabor

        if do_crop:
            if crop_type2:
                generator = Generator_Cropped_type2(c_bits, self.hparams.gen_ns)
            else:
                generator = Generator_Cropped_type1(c_bits, self.hparams.gen_ns)
        else:
            generator = Generator(c_bits, self.hparams.gen_ns)

        self.generator = generator
        self.generator.apply(weights_init)

    def gen_loss_function(self, real_response, gen_response):
        """
        Loss function of the generator. Uses a same-weighted combination of MSE and the SSIM.

        :param real_response: real response of the dataset
        :param gen_response: generated response of the generator
        :return: loss for the prediction
        """
        normalized_real = self.denormalize(real_response)
        normalized_gen = self.denormalize(gen_response)
        mse_criterion = nn.MSELoss()

        ssim_loss = 1 - ssim(normalized_real, normalized_gen)
        mse_loss = mse_criterion(real_response, gen_response)

        return ssim_loss, mse_loss

    def training_step(self, batch, batch_idx):
        real_challenge, real_response = batch
        gen_response = self.generator(real_challenge)

        ssim_loss, mse_loss = self.gen_loss_function(real_response, gen_response)
        loss = ssim_loss + mse_loss

        return {'ssim_loss': ssim_loss,
                'mse_loss': mse_loss,
                'loss': loss}

    def test_step(self, batch, batch_idx):
        challenge, real_response = batch
        gen_response = self.generator(challenge)

        normalized_real = self.denormalize(real_response)
        normalized_gen = self.denormalize(gen_response)

        ssims = calc_ssim(normalized_real, normalized_gen, keep_first_dim=True)
        pear_coeffs = calc_pc(real_response, gen_response, keep_first_dim=True)

        do_crop_responses = not self.crop
        gabor = lambda real, gen: utils.calc_gabor_fhd(real, gen, do_crop_responses, self.img_size, self.crop_size,
                                                       use_custom=self.custom_gabor).item()

        fhds = [gabor(real, gen) for (real, gen) in zip(real_response.cpu(), gen_response.cpu())]

        return {'ssim': ssims, 'pc': pear_coeffs, 'fhd': fhds}

    def test_epoch_end(self, outputs):
        ssim = torch.cat([output["ssim"] for output in outputs]).flatten().tolist()
        pc = torch.cat([output["pc"] for output in outputs]).flatten().tolist()
        fhd = np.array(list(itertools.chain(*[output["fhd"] for output in outputs]))).flatten().tolist()

        return {'FHD': fhd, 'PC': pc, 'SSIM': ssim}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.generator.parameters(), self.hparams.gen_lr,
                                     (self.hparams.gen_beta1, self.hparams.gen_beta2))
        return optimizer
