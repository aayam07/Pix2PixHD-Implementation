import os
import torch

def save_network(net, label, epoch, save_dir):
    filename = f'{label}_epoch_{epoch}.pth'
    path = os.path.join(save_dir, filename)
    torch.save(net.state_dict(), path)

def load_network(net, path, device):
    net.load_state_dict(torch.load(path, map_location=device))
    return net