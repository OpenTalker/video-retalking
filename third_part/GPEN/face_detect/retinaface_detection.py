'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from face_detect.data import cfg_re50
from face_detect.layers.functions.prior_box import PriorBox
from face_detect.utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from face_detect.facemodels.retinaface import RetinaFace
from face_detect.utils.box_utils import decode, decode_landm
import time
import torch.nn.functional as F


class RetinaFaceDetection(object):
    def __init__(self, base_dir, device='cuda', network='RetinaFace-R50'):
        torch.set_grad_enabled(False)
        cudnn.benchmark = True
        self.pretrained_path = os.path.join(base_dir, network+'.pth')
        self.device = device #torch.cuda.current_device()
        self.cfg = cfg_re50
        self.net = RetinaFace(cfg=self.cfg, phase='test')
        self.load_model()
        self.net = self.net.to(device)

        self.mean = torch.tensor([[[[104]], [[117]], [[123]]]]).to(device)

    def check_keys(self, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(self.net.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def load_model(self, load_to_cpu=False):
        #if load_to_cpu:
        #    pretrained_dict = torch.load(self.pretrained_path, map_location=lambda storage, loc: storage)
        #else:
        #    pretrained_dict = torch.load(self.pretrained_path, map_location=lambda storage, loc: storage.cuda())
        pretrained_dict = torch.load(self.pretrained_path, map_location=torch.device('cpu'))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(pretrained_dict)
        self.net.load_state_dict(pretrained_dict, strict=False)
        self.net.eval()
    
    def detect(self, img_raw, resize=1, confidence_threshold=0.9, nms_threshold=0.4, top_k=5000, keep_top_k=750, save_image=False):
        img = np.float32(img_raw)

        im_height, im_width = img.shape[:2]
        ss = 1.0
        # tricky
        if max(im_height, im_width) > 1500:
            ss = 1000.0/max(im_height, im_width)
            img = cv2.resize(img, (0,0), fx=ss, fy=ss)
            im_height, im_width = img.shape[:2]

        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)
        
        with torch.no_grad():
            loc, conf, landms = self.net(img)  # forward pass

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        # sort faces(delete)
        '''
        fscores = [det[4] for det in dets]
        sorted_idx = sorted(range(len(fscores)), key=lambda k:fscores[k], reverse=False) # sort index
        tmp = [landms[idx] for idx in sorted_idx]
        landms = np.asarray(tmp)
        '''
        
        landms = landms.reshape((-1, 5, 2))
        landms = landms.transpose((0, 2, 1))
        landms = landms.reshape(-1, 10, )
        return dets/ss, landms/ss

    def detect_tensor(self, img, resize=1, confidence_threshold=0.9, nms_threshold=0.4, top_k=5000, keep_top_k=750, save_image=False):
        im_height, im_width = img.shape[-2:]
        ss = 1000/max(im_height, im_width)
        img = F.interpolate(img, scale_factor=ss)
        im_height, im_width = img.shape[-2:]
        scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(self.device)
        img -= self.mean

        loc, conf, landms = self.net(img)  # forward pass

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        # sort faces(delete)
        '''
        fscores = [det[4] for det in dets]
        sorted_idx = sorted(range(len(fscores)), key=lambda k:fscores[k], reverse=False) # sort index
        tmp = [landms[idx] for idx in sorted_idx]
        landms = np.asarray(tmp)
        '''
        
        landms = landms.reshape((-1, 5, 2))
        landms = landms.transpose((0, 2, 1))
        landms = landms.reshape(-1, 10, )
        return dets/ss, landms/ss
