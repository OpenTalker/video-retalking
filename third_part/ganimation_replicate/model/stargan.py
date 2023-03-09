import torch
from .base_model import BaseModel
from . import model_utils



class StarGANModel(BaseModel):
    """docstring for StarGANModel"""
    def __init__(self):
        super(StarGANModel, self).__init__()
        self.name = "StarGAN"

    def initialize(self, opt):
        super(StarGANModel, self).initialize(opt)

        self.net_gen = model_utils.define_splitG(self.opt.img_nc, self.opt.aus_nc, self.opt.ngf, use_dropout=self.opt.use_dropout, 
                    norm=self.opt.norm, init_type=self.opt.init_type, init_gain=self.opt.init_gain, gpu_ids=self.gpu_ids)
        self.models_name.append('gen')
        
        if self.is_train:
            self.net_dis = model_utils.define_splitD(self.opt.img_nc, self.opt.aus_nc, self.opt.final_size, self.opt.ndf, 
                    norm=self.opt.norm, init_type=self.opt.init_type, init_gain=self.opt.init_gain, gpu_ids=self.gpu_ids)
            self.models_name.append('dis')

        if self.opt.load_epoch > 0:
            self.load_ckpt(self.opt.load_epoch)

    def setup(self):
        super(StarGANModel, self).setup()
        if self.is_train:
            # setup optimizer
            self.optim_gen = torch.optim.Adam(self.net_gen.parameters(),
                            lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optims.append(self.optim_gen)
            self.optim_dis = torch.optim.Adam(self.net_dis.parameters(), 
                            lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optims.append(self.optim_dis)

            # setup schedulers
            self.schedulers = [model_utils.get_scheduler(optim, self.opt) for optim in self.optims]

    def feed_batch(self, batch):
        self.src_img = batch['src_img'].to(self.device)
        self.tar_aus = batch['tar_aus'].type(torch.FloatTensor).to(self.device)
        if self.is_train:
            self.src_aus = batch['src_aus'].type(torch.FloatTensor).to(self.device)
            self.tar_img = batch['tar_img'].to(self.device)

    def forward(self):
        # generate fake image
        self.fake_img, _, _ = self.net_gen(self.src_img, self.tar_aus)

        # reconstruct real image
        if self.is_train:
            self.rec_real_img, _, _ = self.net_gen(self.fake_img, self.src_aus)

    def backward_dis(self):
        # real image
        pred_real, self.pred_real_aus = self.net_dis(self.src_img)
        self.loss_dis_real = self.criterionGAN(pred_real, True)
        self.loss_dis_real_aus = self.criterionMSE(self.pred_real_aus, self.src_aus)

        # fake image, detach to stop backward to generator
        pred_fake, _ = self.net_dis(self.fake_img.detach()) 
        self.loss_dis_fake = self.criterionGAN(pred_fake, False)

        # combine dis loss
        self.loss_dis =   self.opt.lambda_dis * (self.loss_dis_fake + self.loss_dis_real) \
                        + self.opt.lambda_aus * self.loss_dis_real_aus
        if self.opt.gan_type == 'wgan-gp':
            self.loss_dis_gp = self.gradient_penalty(self.src_img, self.fake_img)
            self.loss_dis = self.loss_dis + self.opt.lambda_wgan_gp * self.loss_dis_gp
        
        # backward discriminator loss
        self.loss_dis.backward()

    def backward_gen(self):
        # original to target domain, should fake the discriminator
        pred_fake, self.pred_fake_aus = self.net_dis(self.fake_img)
        self.loss_gen_GAN = self.criterionGAN(pred_fake, True)
        self.loss_gen_fake_aus = self.criterionMSE(self.pred_fake_aus, self.tar_aus)

        # target to original domain reconstruct, identity loss
        self.loss_gen_rec = self.criterionL1(self.rec_real_img, self.src_img)

        # combine and backward G loss
        self.loss_gen =   self.opt.lambda_dis * self.loss_gen_GAN \
                        + self.opt.lambda_aus * self.loss_gen_fake_aus \
                        + self.opt.lambda_rec * self.loss_gen_rec 

        self.loss_gen.backward()

    def optimize_paras(self, train_gen):
        self.forward()
        # update discriminator
        self.set_requires_grad(self.net_dis, True)
        self.optim_dis.zero_grad()
        self.backward_dis()
        self.optim_dis.step()

        # update G if needed
        if train_gen:
            self.set_requires_grad(self.net_dis, False)
            self.optim_gen.zero_grad()
            self.backward_gen()
            self.optim_gen.step()

    def save_ckpt(self, epoch):
        # save the specific networks
        save_models_name = ['gen', 'dis']
        return super(StarGANModel, self).save_ckpt(epoch, save_models_name)

    def load_ckpt(self, epoch):
        # load the specific part of networks
        load_models_name = ['gen']
        if self.is_train:
            load_models_name.extend(['dis'])
        return super(StarGANModel, self).load_ckpt(epoch, load_models_name)

    def clean_ckpt(self, epoch):
        # load the specific part of networks
        load_models_name = ['gen', 'dis']
        return super(StarGANModel, self).clean_ckpt(epoch, load_models_name)

    def get_latest_losses(self):
        get_losses_name = ['dis_fake', 'dis_real', 'dis_real_aus', 'gen_rec']
        return super(StarGANModel, self).get_latest_losses(get_losses_name)

    def get_latest_visuals(self):
        visuals_name = ['src_img', 'tar_img', 'fake_img']
        if self.is_train:
            visuals_name.extend(['rec_real_img'])
        return super(StarGANModel, self).get_latest_visuals(visuals_name)
