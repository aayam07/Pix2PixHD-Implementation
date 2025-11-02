import random

class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []
    def query(self, image):
        if self.pool_size==0:
            return image
        if len(self.images) < self.pool_size:
            self.images.append(image.detach())
            return image
        if random.random() > 0.5:
            idx = random.randint(0, self.pool_size-1)
            tmp = self.images[idx].clone()
            self.images[idx] = image.detach()
            return tmp
        else:
            return image