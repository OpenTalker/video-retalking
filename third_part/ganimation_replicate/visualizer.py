import os
import numpy as np
import torch
import math
from PIL import Image
# import matplotlib.pyplot as plt



class Visualizer(object):
    """docstring for Visualizer"""
    def __init__(self):
        super(Visualizer, self).__init__()

    def initialize(self, opt):
        self.opt = opt
        # self.vis_saved_dir = os.path.join(self.opt.ckpt_dir, 'vis_pics')
        # if not os.path.isdir(self.vis_saved_dir):
        #     os.makedirs(self.vis_saved_dir)
        # plt.switch_backend('agg')

        self.display_id = self.opt.visdom_display_id
        if self.display_id > 0:
            import visdom 
            self.ncols = 8
            self.vis = visdom.Visdom(server="http://localhost", port=self.opt.visdom_port, env=self.opt.visdom_env)

    def throw_visdom_connection_error(self):
        print('\n\nno visdom server.')
        exit(1)

    def print_losses_info(self, info_dict):
        msg = '[{}][Epoch: {:0>3}/{:0>3}; Images: {:0>4}/{:0>4}; Time: {:.3f}s/Batch({}); LR: {:.7f}] '.format(
                self.opt.name, info_dict['epoch'], info_dict['epoch_len'], 
                info_dict['epoch_steps'], info_dict['epoch_steps_len'], 
                info_dict['step_time'], self.opt.batch_size, info_dict['cur_lr'])
        for k, v in info_dict['losses'].items():
            msg += '| {}: {:.4f} '.format(k, v)
        msg += '|'
        print(msg)
        with open(info_dict['log_path'], 'a+') as f:
            f.write(msg + '\n')

    def display_current_losses(self, epoch, counter_ratio, losses_dict):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses_dict.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses_dict[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.opt.name + ' loss over time',
                    'legend':self.plot_data['legend'],
                    'xlabel':'epoch',
                    'ylabel':'loss'},
                win=self.display_id)
        except ConnectionError:
            self.throw_visdom_connection_error()

    def display_online_results(self, visuals, epoch):
        win_id = self.display_id + 24
        images = []
        labels = []
        for label, image in visuals.items():
            if 'mask' in label:  # or 'focus' in label:
                image = (image - 0.5) / 0.5   # convert map from [0, 1] to [-1, 1]
            image_numpy = self.tensor2im(image)
            images.append(image_numpy.transpose([2, 0, 1]))
            labels.append(label)
        try:
            title = ' || '.join(labels)
            self.vis.images(images, nrow=self.ncols, win=win_id,
                            padding=5, opts=dict(title=title))
        except ConnectionError:
            self.throw_visdom_connection_error()
        
    # utils
    def tensor2im(self, input_image, imtype=np.uint8):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        im = self.numpy2im(image_numpy, imtype).resize((80, 80), Image.ANTIALIAS)
        return np.array(im)
        
    def numpy2im(self, image_numpy, imtype=np.uint8):
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))  
        # input should be [0, 1]
        #image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) / 2. + 0.5) * 255.0
        # print(image_numpy.shape)
        image_numpy = image_numpy.astype(imtype)
        im = Image.fromarray(image_numpy)
        # im = Image.fromarray(image_numpy).resize((64, 64), Image.ANTIALIAS)
        return im   # np.array(im)





