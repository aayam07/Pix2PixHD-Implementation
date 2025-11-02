import os
from options.base_options import BaseOptions
from data.dataset import PairedDataset
from models.pix2pix_models import Pix2PixHDModel
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch

if __name__=='__main__':
    opt = BaseOptions().parse()
    opt.mode='test'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PairedDataset(opt.dataroot, phase='test', load_size=opt.load_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = Pix2PixHDModel(opt, device)
    # load latest or named epoch
    ckpt_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if opt.which_epoch=='latest':
        files = [f for f in os.listdir(ckpt_dir) if f.startswith('netG')]
        files.sort()
        ckpt = os.path.join(ckpt_dir, files[-1])
    else:
        ckpt = os.path.join(ckpt_dir, opt.which_epoch)

    model.netG.load_state_dict(torch.load(ckpt,map_location=device))
    model.netG.to(device).eval()
    out_dir = os.path.join('./results', opt.name)
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for i, data in enumerate(loader):
            A = data['A'].to(device)
            B = data['B'].to(device)
            fake = model.netG(A)
            out = torch.cat([A, fake, B], dim=3)
            save_image((out+1)/2, os.path.join(out_dir, f'{i:04d}.png'))
    print('Results saved at', out_dir)