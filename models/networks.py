import torch
import torch.nn as nn
import torch.nn.functional as F

# (1) weight init

def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and classname.find('Conv')!=-1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm')!=-1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# (2) generator: global + local 

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim,dim,3), nn.InstanceNorm2d(dim), nn.ReLU(True), nn.ReflectionPad2d(1), nn.Conv2d(dim,dim,3), nn.InstanceNorm2d(dim))

    def forward(self, x):
        return x + self.conv(x)

class GlobalGenerator(nn.Module):
        def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=6):
            super().__init__()
            model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
            # down
            n_down = 4
            mult = 1
            for i in range(n_down):
                prev = mult
                mult *= 2
                model += [nn.Conv2d(ngf*prev, ngf*mult, 3, stride=2, padding=1), nn.InstanceNorm2d(ngf*mult), nn.ReLU(True)]
            for i in range(n_blocks):
                model += [ResnetBlock(ngf*mult)]
            # up
            for i in range(n_down):
                prev = mult
                mult //= 2
                model += [nn.ConvTranspose2d(ngf*prev, ngf*mult, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(ngf*mult), nn.ReLU(True)]
            
            model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7), nn.Tanh()]
            self.model = nn.Sequential(*model)
        
        def forward(self, x):
            return self.model(x)


class LocalEnhancer(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=32, n_blocks=3):
        super().__init__()
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        
        for i in range(n_blocks):
            model += [ResnetBlock(ngf)]
        
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)
    
    def forward(self, x, global_feat):
        g = F.interpolate(global_feat, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return self.model(x + g)


class CoarseToFineGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.globalG = GlobalGenerator()
        self.local = LocalEnhancer()
    
    def forward(self, x):
        g = self.globalG(x)
        l = self.local(x, g)
        return 0.5*(g+l)

# discriminator
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=6, ndf=64, n_layers=3):
        super().__init__()
        kw=4; padw=1
        sequence = [nn.Conv2d(input_nc, ndf, kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kw, stride=2, padding=padw), nn.InstanceNorm2d(ndf*nf_mult), nn.LeakyReLU(0.2, True)]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kw, stride=1, padding=padw), nn.InstanceNorm2d(ndf*nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf*nf_mult, 1, kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        return self.model(x)


class MultiscaleDiscriminator(nn.Module):
    
    def __init__(self, n_discriminators=3):
        super().__init__()
        self.n = n_discriminators
        for i in range(self.n):
            net = NLayerDiscriminator()
            setattr(self, 'disc_%d'%i, net)
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, x):
        results = []
        input_down = x
        for i in range(self.n):
            net = getattr(self, 'disc_%d'%i)
            results.append(net(input_down))
            input_down = self.downsample(input_down)
        return results