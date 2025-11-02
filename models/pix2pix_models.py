from .networks import CoarseToFineGenerator, MultiscaleDiscriminator, weights_init
from .losses import GANLoss, VGGLoss
import torch

class Pix2PixHDModel:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.netG = CoarseToFineGenerator().to(device)
        self.netD = MultiscaleDiscriminator().to(device)
        self.netG.apply(weights_init)
        for i in range(self.netD.n):
            getattr(self.netD, 'disc_%d'%i).apply(weights_init)
        self.gan_loss = GANLoss()
        self.vgg_loss = VGGLoss(device)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(0.5,0.999))
        d_params = []
        for i in range(self.netD.n):
            d_params += list(getattr(self.netD, 'disc_%d'%i).parameters())
        self.optimizer_D = torch.optim.Adam(d_params, lr=opt.lr, betas=(0.5,0.999))

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        fake_AB = torch.cat([self.real_A, self.fake_B.detach()], dim=1)
        real_AB = torch.cat([self.real_A, self.real_B], dim=1)
        pred_fake = self.netD(fake_AB)
        pred_real = self.netD(real_AB)
        loss_D = 0
        for pf, pr in zip(pred_fake, pred_real):
            loss_D += self.gan_loss(pf, False) + self.gan_loss(pr, True)
        loss_D.backward()
        return loss_D

    def backward_G(self):
        fake_AB = torch.cat([self.real_A, self.fake_B], dim=1)
        pred_fake = self.netD(fake_AB)
        loss_G_GAN = 0
        for pf in pred_fake:
            loss_G_GAN += self.gan_loss(pf, True)
        loss_vgg = self.vgg_loss(self.fake_B, self.real_B)
        loss_G = loss_G_GAN + 10.0 * loss_vgg
        loss_G.backward()
        return loss_G

def optimize_parameters(self):
    # forward
    self.forward()
    # D
    self.netD.zero_grad()
    loss_D = self.backward_D()
    self.optimizer_D.step()
    8
    # G
    self.netG.zero_grad()
    loss_G = self.backward_G()
    self.optimizer_G.step()
    return loss_D, loss_G