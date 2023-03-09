'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import os
import cv2
import torch
import numpy as np
from face_parse.parse_model import ParseNet
import torch.nn.functional as F

from face_parse.model import BiSeNet
import torchvision.transforms as transforms

class FaceParse(object):
    def __init__(self, base_dir='./', model='ParseNet-latest', device='cuda', mask_map = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0]):
        self.mfile = os.path.join(base_dir, model+'.pth')
        self.size = 512
        self.device = device

        '''
        0: 'background' 1: 'skin'   2: 'nose'
        3: 'eye_g'  4: 'l_eye'  5: 'r_eye'
        6: 'l_brow' 7: 'r_brow' 8: 'l_ear'
        9: 'r_ear'  10: 'mouth' 11: 'u_lip'
        12: 'l_lip' 13: 'hair'  14: 'hat'
        15: 'ear_r' 16: 'neck_l'    17: 'neck'
        18: 'cloth'
        '''
        # self.MASK_COLORMAP = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
        #self.#MASK_COLORMAP = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]] = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [0, 0, 0], [0, 0, 0]]
        # self.MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
        self.MASK_COLORMAP = mask_map

        self.load_model()

    def load_model(self):
        self.faceparse = ParseNet(self.size, self.size, 32, 64, 19, norm_type='bn', relu_type='LeakyReLU', ch_range=[32, 256])
        self.faceparse.load_state_dict(torch.load(self.mfile, map_location=torch.device('cpu')))
        self.faceparse.to(self.device)
        self.faceparse.eval()

    def process(self, im, masks=[0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0]):
        im = cv2.resize(im, (self.size, self.size))
        imt = self.img2tensor(im)
        with torch.no_grad():
            pred_mask, sr_img_tensor = self.faceparse(imt)  # (1, 19, 512, 512)
        mask = self.tenor2mask(pred_mask, masks)

        return mask

    def process_tensor(self, imt):
        imt = F.interpolate(imt.flip(1)*2-1, (self.size, self.size))
        pred_mask, sr_img_tensor = self.faceparse(imt)

        mask = pred_mask.argmax(dim=1)
        for idx, color in enumerate(self.MASK_COLORMAP):
            mask = torch.where(mask==idx, color, mask)
        #mask = mask.repeat(3, 1, 1).unsqueeze(0) #.cpu().float().numpy()
        mask = mask.unsqueeze(0)

        return mask

    def img2tensor(self, img):
        img = img[..., ::-1] # BGR to RGB
        img = img / 255. * 2 - 1
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return img_tensor.float()

    def tenor2mask(self, tensor, masks):
        if len(tensor.shape) < 4:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[1] > 1:
            tensor = tensor.argmax(dim=1) 

        tensor = tensor.squeeze(1).data.cpu().numpy()   # (1, 512, 512)
        color_maps = []
        for t in tensor:
            #tmp_img = np.zeros(tensor.shape[1:] + (3,))
            tmp_img = np.zeros(tensor.shape[1:])
            for idx, color in enumerate(masks):
                tmp_img[t == idx] = color
            color_maps.append(tmp_img.astype(np.uint8))
        return color_maps



class FaceParse_v2(object):
    def __init__(self, device='cuda', mask_map = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0]):
        self.mfile = '/apdcephfs/private_quincheng/Expression/face-parsing.PyTorch/res/cp/79999_iter.pth'
        self.size = 512
        self.device = device

        '''
        0: 'background' 1: 'skin'   2: 'nose'
        3: 'eye_g'  4: 'l_eye'  5: 'r_eye'
        6: 'l_brow' 7: 'r_brow' 8: 'l_ear'
        9: 'r_ear'  10: 'mouth' 11: 'u_lip'
        12: 'l_lip' 13: 'hair'  14: 'hat'
        15: 'ear_r' 16: 'neck_l'    17: 'neck'
        18: 'cloth'
        '''
        # self.MASK_COLORMAP = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
        #self.#MASK_COLORMAP = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]] = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [0, 0, 0], [0, 0, 0]]
        # self.MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
        self.MASK_COLORMAP = mask_map
        self.load_model()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def load_model(self):
        self.faceparse = BiSeNet(n_classes=19)
        self.faceparse.load_state_dict(torch.load(self.mfile))
        self.faceparse.to(self.device)
        self.faceparse.eval()

    def process(self, im, masks=[0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0]):
        im = cv2.resize(im[...,::-1], (self.size, self.size))
        im = self.to_tensor(im)
        imt = torch.unsqueeze(im, 0).to(self.device)
        with torch.no_grad():
            pred_mask = self.faceparse(imt)[0]
        mask = self.tenor2mask(pred_mask, masks)
        return mask

    # def img2tensor(self, img):
    #     img = img[..., ::-1] # BGR to RGB
    #     img = img / 255. * 2 - 1
    #     img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
    #     return img_tensor.float()

    def tenor2mask(self, tensor, masks):
        if len(tensor.shape) < 4:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[1] > 1:
            tensor = tensor.argmax(dim=1) 

        tensor = tensor.squeeze(1).data.cpu().numpy()
        color_maps = []
        for t in tensor:
            #tmp_img = np.zeros(tensor.shape[1:] + (3,))
            tmp_img = np.zeros(tensor.shape[1:])
            for idx, color in enumerate(masks):
                tmp_img[t == idx] = color
            color_maps.append(tmp_img.astype(np.uint8))
        return color_maps