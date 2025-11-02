import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PairedDataset(Dataset):

    def __init__(self, root, phase='train', load_size=512, transform=None, input_dir='input', target_dir='target'):
        
        folder = os.path.join(root, phase)
        files = sorted(glob(os.path.join(folder,'*')))

        if len(files)>0:
            # assuming concatenated
            self.concatenated = True
            self.files = files
        else:
            self.concatenated = False
            in_dir = os.path.join(folder, input_dir)
            tar_dir = os.path.join(folder, target_dir)
            in_files = sorted(glob(os.path.join(in_dir,'*')))
            tar_files = sorted(glob(os.path.join(tar_dir,'*')))

            if len(in_files)==0:
                raise RuntimeError('No input files found')
            
            self.files = list(zip(in_files, tar_files))
        self.transform = transform or transforms.Compose([transforms.Resize((load_size,load_size)), transforms.ToTensor()])

    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.concatenated:
            path = self.files[idx]
            img = Image.open(path).convert('RGB')
            w,h = img.size
            left = img.crop((0,0,w//2,h))
            right = img.crop((w//2,0,w,h))
            A = self.transform(left)
            B = self.transform(right)   
        else:
            a,b = self.files[idx]
            A = self.transform(Image.open(a).convert('RGB'))
            B = self.transform(Image.open(b).convert('RGB'))
        return {'A':A, 'B':B}

    
