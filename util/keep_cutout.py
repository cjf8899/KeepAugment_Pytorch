import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F


class Keep_Cutout_Low(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, train_transform, mean, std, length, early=False):
        self.trans = train_transform
        self.length = int(length/2)
        self.early = early
        self.denomal = transforms.Compose([
            transforms.Normalize((-mean[0]/std[0],-mean[1]/std[1], -mean[2]/std[2]), (1/std[0], 1/std[1], 1/std[2])),
            transforms.ToPILImage()
        ])

    def __call__(self, images, model):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        images_ = images.clone().detach()
        images_half = F.interpolate(images_, scale_factor=0.5, mode='bicubic',align_corners=True)
        images_half.requires_grad = True

        if self.early:
            preds = model(images_half,True)
        else:
            preds = model(images_half)
        
        score, _ = torch.max(preds, 1)
        score.mean().backward()
        slc_, _ = torch.max(torch.abs(images_half.grad), dim=1)
        
#         slc_ = torch.unsqueeze(slc_,dim=1)
#         slc_ = F.upsample(slc_, size=(slc_.shape[-2]*2,slc_.shape[-1]*2), mode='bicubic')
#         slc_ = torch.squeeze(slc_,dim=1)
        
        
        b,h,w = slc_.shape
        slc_ = slc_.view(slc_.size(0), -1)
        slc_ -= slc_.min(1, keepdim=True)[0]
        slc_ /= slc_.max(1, keepdim=True)[0]
        slc_ = slc_.view(b, h, w)
        
        
        for i,(img, slc) in enumerate(zip(images_, slc_)):
            mask = np.ones((h*2, w*2), np.float32)
            while(True):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)
                if slc[y1: y2, x1: x2].mean() < 0.6:
                    mask[y1*2: y2*2, x1*2: x2*2] = 0.
                    break

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img).cuda()
            img = img * mask
            images[i] = self.trans(self.denomal(img))
                
        model.train()
        for param in model.parameters():
            param.requires_grad = True
#         for img in images:
#             save_image(img, 'test.jpg')
        return images.cuda()


class Keep_Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, train_transform, mean, std, length, early=False):
        self.trans = train_transform
        self.length = length
        self.early = early
        self.denomal = transforms.Compose([
            transforms.Normalize((-mean[0]/std[0],-mean[1]/std[1], -mean[2]/std[2]), (1/std[0], 1/std[1], 1/std[2])),
            transforms.ToPILImage()
        ])
        
    def __call__(self, images, model):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        images_ = images.clone().detach()
        images_.requires_grad = True

        if self.early:
            preds = model(images_,True)
        else:
            preds = model(images_)

        score, _ = torch.max(preds, 1)
        score.mean().backward()
        slc_, _ = torch.max(torch.abs(images_.grad), dim=1)
        
        b,h,w = slc_.shape
        
        slc_ = slc_.view(slc_.size(0), -1)
        slc_ -= slc_.min(1, keepdim=True)[0]
        slc_ /= slc_.max(1, keepdim=True)[0]
        slc_ = slc_.view(b, h, w)
        
        for i,(img, slc) in enumerate(zip(images_, slc_)):
            mask = np.ones((h, w), np.float32)
            while(True):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                if slc[y1: y2, x1: x2].mean() < 0.6:
                    mask[y1: y2, x1: x2] = 0.
                    break

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img).cuda()
            img = img * mask
            images[i] = self.trans(self.denomal(img))
                
        model.train()
        for param in model.parameters():
            param.requires_grad = True
            
#         for img in images:
#             save_image(img, 'test.jpg')
        return images.cuda()

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
