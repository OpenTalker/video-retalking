import argparse
import torch
import os
from datetime import datetime
import time
import torch 
import random
import numpy as np 
import sys



class Options(object):
    """docstring for Options"""
    def __init__(self):
        super(Options, self).__init__()
        
    def initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--mode', type=str, default='train', help='Mode of code. [train|test]')
        parser.add_argument('--model', type=str, default='ganimation', help='[ganimation|stargan], see model.__init__ from more details.')
        parser.add_argument('--lucky_seed', type=int, default=0, help='seed for random initialize, 0 to use current time.')
        parser.add_argument('--visdom_env', type=str, default="main", help='visdom env.')
        parser.add_argument('--visdom_port', type=int, default=8097, help='visdom port.')
        parser.add_argument('--visdom_display_id', type=int, default=1, help='set value larger than 0 to display with visdom.')
        
        parser.add_argument('--results', type=str, default="results", help='save test results to this path.')
        parser.add_argument('--interpolate_len', type=int, default=5, help='interpolate length for test.')
        parser.add_argument('--no_test_eval', action='store_true', help='do not use eval mode during test time.')
        parser.add_argument('--save_test_gif', action='store_true', help='save gif images instead of the concatenation of static images.')

        parser.add_argument('--data_root', required=False, help='paths to data set.')
        parser.add_argument('--imgs_dir', type=str, default="imgs", help='path to image')
        parser.add_argument('--aus_pkl', type=str, default="aus_openface.pkl", help='AUs pickle dictionary.')
        parser.add_argument('--train_csv', type=str, default="train_ids.csv", help='train images paths')
        parser.add_argument('--test_csv', type=str, default="test_ids.csv", help='test images paths')

        parser.add_argument('--batch_size', type=int, default=25, help='input batch size.')
        parser.add_argument('--serial_batches', action='store_true', help='if specified, input images in order.')
        parser.add_argument('--n_threads', type=int, default=6, help='number of workers to load data.')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='maximum number of samples.')

        parser.add_argument('--resize_or_crop', type=str, default='none', help='Preprocessing image, [resize_and_crop|crop|none]')
        parser.add_argument('--load_size', type=int, default=148, help='scale image to this size.')
        parser.add_argument('--final_size', type=int, default=128, help='crop image to this size.')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip image.')
        parser.add_argument('--no_aus_noise', action='store_true', help='if specified, add noise to target AUs.')

        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids, eg. 0,1,2; -1 for cpu.')
        parser.add_argument('--ckpt_dir', type=str, default='./ckpts', help='directory to save check points.')
        parser.add_argument('--load_epoch', type=int, default=0, help='load epoch; 0: do not load')
        parser.add_argument('--log_file', type=str, default="logs.txt", help='log loss')
        parser.add_argument('--opt_file', type=str, default="opt.txt", help='options file')

        # train options 
        parser.add_argument('--img_nc', type=int, default=3, help='image number of channel')
        parser.add_argument('--aus_nc', type=int, default=17, help='aus number of channel')
        parser.add_argument('--ngf', type=int, default=64, help='ngf')
        parser.add_argument('--ndf', type=int, default=64, help='ndf')
        parser.add_argument('--use_dropout', action='store_true', help='if specified, use dropout.')
        
        parser.add_argument('--gan_type', type=str, default='wgan-gp', help='GAN loss [wgan-gp|lsgan|gan]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [batch|instance|none]')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--niter', type=int, default=20, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=10, help='# of iter to linearly decay learning rate to zero')
        
        # loss options 
        parser.add_argument('--lambda_dis', type=float, default=1.0, help='discriminator weight in loss')
        parser.add_argument('--lambda_aus', type=float, default=160.0, help='AUs weight in loss')
        parser.add_argument('--lambda_rec', type=float, default=10.0, help='reconstruct loss weight')
        parser.add_argument('--lambda_mask', type=float, default=0, help='mse loss weight')
        parser.add_argument('--lambda_tv', type=float, default=0, help='total variation loss weight')
        parser.add_argument('--lambda_wgan_gp', type=float, default=10., help='wgan gradient penalty weight')

        # frequency options
        parser.add_argument('--train_gen_iter', type=int, default=5, help='train G every n iterations.')
        parser.add_argument('--print_losses_freq', type=int, default=100, help='print log every print_freq step.')
        parser.add_argument('--plot_losses_freq', type=int, default=20000, help='plot log every plot_freq step.')
        parser.add_argument('--sample_img_freq', type=int, default=2000, help='draw image every sample_img_freq step.')
        parser.add_argument('--save_epoch_freq', type=int, default=2, help='save checkpoint every save_epoch_freq epoch.')
        
        return parser

    def parse(self):
        parser = self.initialize()
        parser.set_defaults(name=datetime.now().strftime("%y%m%d_%H%M%S"))
        opt = parser.parse_args()

        dataset_name = os.path.basename(opt.data_root.strip('/'))
        # update checkpoint dir
        if opt.mode == 'train' and opt.load_epoch == 0:
            opt.ckpt_dir = os.path.join(opt.ckpt_dir, dataset_name, opt.model, opt.name)
            if not os.path.exists(opt.ckpt_dir):
                os.makedirs(opt.ckpt_dir)

        # if test, disable visdom, update results path
        if opt.mode == "test":
            opt.visdom_display_id = 0
            opt.results = os.path.join(opt.results, "%s_%s_%s" % (dataset_name, opt.model, opt.load_epoch))
            if not os.path.exists(opt.results):
                os.makedirs(opt.results)

        # set gpu device
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            cur_id = int(str_id)
            if cur_id >= 0:
                opt.gpu_ids.append(cur_id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # set seed 
        if opt.lucky_seed == 0:
            opt.lucky_seed = int(time.time())
        random.seed(a=opt.lucky_seed)
        np.random.seed(seed=opt.lucky_seed)
        torch.manual_seed(opt.lucky_seed)
        if len(opt.gpu_ids) > 0:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(opt.lucky_seed)
            torch.cuda.manual_seed_all(opt.lucky_seed)
            
        # write command to file
        script_dir = opt.ckpt_dir 
        with open(os.path.join(os.path.join(script_dir, "run_script.sh")), 'a+') as f:
            f.write("[%5s][%s]python %s\n" % (opt.mode, opt.name, ' '.join(sys.argv)))

        # print and write options file
        msg = ''
        msg += '------------------- [%5s][%s]Options --------------------\n' % (opt.mode, opt.name)
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default_v = parser.get_default(k)
            if v != default_v:
                comment = '\t[default: %s]' % str(default_v)
            msg += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        msg += '--------------------- [%5s][%s]End ----------------------\n' % (opt.mode, opt.name)
        print(msg)
        with open(os.path.join(os.path.join(script_dir, "opt.txt")), 'a+') as f:
            f.write(msg + '\n\n')

        return opt






