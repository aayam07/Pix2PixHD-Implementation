import os
from options.base_options import BaseOptions
from data.dataset import PairedDataset
from models.pix2pix_models import Pix2PixHDModel
from torch.utils.data import DataLoader
import torch

if __name__=='__main__':
    opt = BaseOptions().parse()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PairedDataset(opt.dataroot, phase='train',
    load_size=opt.load_size)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
    num_workers=4, drop_last=True)
    model = Pix2PixHDModel(opt, device)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(opt.n_epochs):
        for i, data in enumerate(loader):
            model.set_input(data)
            loss_D, loss_G = model.optimize_parameters()
            if i % 10 == 0:
                print(f'Epoch {epoch} Iter {i} Loss_D {loss_D.item():.4f} Loss_G {loss_G.item():.4f}')
        # save
        from util.util import save_network
        save_network(model.netG, 'netG', epoch, save_dir)
        save_network(model.netD, 'netD', epoch, save_dir)