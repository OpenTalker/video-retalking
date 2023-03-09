import torch
import os
from collections import OrderedDict
import random
from . import model_utils


class BaseModel:
    """docstring for BaseModel"""
    def __init__(self):
        super(BaseModel, self).__init__()
        self.name = "Base"

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = self.opt.gpu_ids
        self.device = torch.device('cuda:%d' % self.gpu_ids[0] if self.gpu_ids else 'cpu')
        self.is_train = self.opt.mode == "train"
        # inherit to define network model 
        self.models_name = []
        
    def setup(self):
        # print("%s with Model [%s]" % (self.opt.mode.capitalize(), self.name))
        if self.is_train:
            self.set_train()
            # define loss function
            self.criterionGAN = model_utils.GANLoss(gan_type=self.opt.gan_type).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionMSE = torch.nn.MSELoss().to(self.device)
            self.criterionTV = model_utils.TVLoss().to(self.device)
            torch.nn.DataParallel(self.criterionGAN, self.gpu_ids)
            torch.nn.DataParallel(self.criterionL1, self.gpu_ids)
            torch.nn.DataParallel(self.criterionMSE, self.gpu_ids)
            torch.nn.DataParallel(self.criterionTV, self.gpu_ids)
            # inherit to set up train/val/test status
            self.losses_name = []
            self.optims = []
            self.schedulers = []
        else:
            self.set_eval()

    def set_eval(self):
        print("Set model to Test state.")
        for name in self.models_name:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                if True:
                    net.eval()
                    print("Set net_%s to EVAL." % name)
                else:
                    net.train()
        self.is_train = False

    def set_train(self):
        print("Set model to Train state.")
        for name in self.models_name:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.train()
                print("Set net_%s to TRAIN." % name)
        self.is_train = True

    def set_requires_grad(self, parameters, requires_grad=False):
        if not isinstance(parameters, list):
            parameters = [parameters]
        for param in parameters:
            if param is not None:
                param.requires_grad = requires_grad

    def get_latest_visuals(self, visuals_name):
        visual_ret = OrderedDict()
        for name in visuals_name:
            if isinstance(name, str) and hasattr(self, name):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_latest_losses(self, losses_name):
        errors_ret = OrderedDict()
        for name in losses_name:
            if isinstance(name, str):
                cur_loss = float(getattr(self, 'loss_' + name))
                # cur_loss_lambda = 1. if len(losses_name) == 1 else float(getattr(self.opt, 'lambda_' + name))
                # errors_ret[name] = cur_loss * cur_loss_lambda
                errors_ret[name] = cur_loss
        return errors_ret

    def feed_batch(self, batch):
        pass 

    def forward(self):
        pass

    def optimize_paras(self):
        pass

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optims[0].param_groups[0]['lr']
        return lr

    def save_ckpt(self, epoch, models_name):
        for name in models_name:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.opt.ckpt_dir, save_filename)
                net = getattr(self, 'net_' + name)
                # save cpu params, so that it can be used in other GPU settings
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.to(self.gpu_ids[0])
                    net = torch.nn.DataParallel(net, self.gpu_ids)
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_ckpt(self, epoch, models_name):
        # print(models_name)
        for name in models_name:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                # load_path = os.path.join(self.opt.ckpt_dir, load_filename)
                # assert os.path.isfile(load_path), "File '%s' does not exist." % load_path
                
                # pretrained_state_dict = torch.load(load_path, map_location=str(self.device))
                pretrained_state_dict = torch.load('checkpoints/30_net_gen.pth', map_location=str('cuda:0'))
                if hasattr(pretrained_state_dict, '_metadata'):
                    del pretrained_state_dict._metadata

                net = getattr(self, 'net_' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                # load only existing keys
                pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in net.state_dict()}
                # for k, v in pretrained_state_dict.items():
                #     print(k)
                # assert False
                net.load_state_dict(pretrained_dict)
                print("[Info] Successfully load trained weights for net_%s." % name)

    def clean_ckpt(self, epoch, models_name):
        for name in models_name:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.opt.ckpt_dir, load_filename)
                if os.path.isfile(load_path):
                    os.remove(load_path)

    def gradient_penalty(self, input_img, generate_img):
        # interpolate sample
        alpha = torch.rand(input_img.size(0), 1, 1, 1).to(self.device)
        inter_img = (alpha * input_img.data + (1 - alpha) * generate_img.data).requires_grad_(True)
        inter_img_prob, _ = self.net_dis(inter_img)

        # computer gradient penalty: x: inter_img, y: inter_img_prob
        # (L2_norm(dy/dx) - 1)**2
        dydx = torch.autograd.grad(outputs=inter_img_prob,
                                   inputs=inter_img,
                                   grad_outputs=torch.ones(inter_img_prob.size()).to(self.device),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2) 



