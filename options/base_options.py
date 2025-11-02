import argparse

class BaseOptions:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--dataroot', required=True, help='path to dataset root')
        self.parser.add_argument('--name', type=str, default='experiment')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
        self.parser.add_argument('--batch_size', type=int, default=4)
        self.parser.add_argument('--load_size', type=int, default=512)
        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--n_epochs', type=int, default=100)
        self.parser.add_argument('--n_epochs_decay', type=int, default=100)
        self.parser.add_argument('--gpu_ids', type=str, default='0')
        self.parser.add_argument('--which_epoch', type=str, default='latest')
        self.parser.add_argument('--mode', type=str, default='train', choices=['train','test'])

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt