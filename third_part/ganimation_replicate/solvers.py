# Created on Dec 13, 2018
# @author: Yuedong Chen

from data import create_dataloader
from model import create_model
from visualizer import Visualizer
import time
import os
import torch
import numpy as np
from PIL import Image
import imageio  # Added import for imageio

class Solver:
    def __init__(self):
        self.opt = None
        self.visual = Visualizer()

    def initialize(self, opt):
        self.opt = opt
        self.visual.initialize(self.opt)

    def run_solver(self):
        if self.opt.mode == "train":
            self.train_networks()
        else:
            self.test_networks(self.opt)

    def train_networks(self):
        self.init_train_setting()

        for epoch in range(self.opt.epoch_count, self.epoch_len + 1):
            self.train_epoch(epoch)
            self.cur_lr = self.train_model.update_learning_rate()
            if epoch % self.opt.save_epoch_freq == 0:
                self.train_model.save_ckpt(epoch)

        self.train_model.save_ckpt(self.epoch_len)

    def init_train_setting(self):
        self.train_dataset = create_dataloader(self.opt)
        self.train_model = create_model(self.opt)
        self.train_total_steps = 0
        self.epoch_len = self.opt.niter + self.opt.niter_decay
        self.cur_lr = self.opt.lr

    def train_epoch(self, epoch):
        epoch_steps = 0
        last_print_step_t = time.time()

        for idx, batch in enumerate(self.train_dataset):
            self.train_total_steps += self.opt.batch_size
            epoch_steps += self.opt.batch_size
            self.train_model.feed_batch(batch)
            self.train_model.optimize_paras(train_gen=(idx % self.opt.train_gen_iter == 0))

            if self.train_total_steps % self.opt.print_losses_freq == 0:
                cur_losses = self.train_model.get_latest_losses()
                avg_step_t = (time.time() - last_print_step_t) / self.opt.print_losses_freq
                last_print_step_t = time.time()
                info_dict = {'epoch': epoch, 'epoch_len': self.epoch_len,
                             'epoch_steps': idx * self.opt.batch_size, 'epoch_steps_len': len(self.train_dataset),
                             'step_time': avg_step_t, 'cur_lr': self.cur_lr,
                             'log_path': os.path.join(self.opt.ckpt_dir, self.opt.log_file),
                             'losses': cur_losses
                             }
                self.visual.print_losses_info(info_dict)

            if self.train_total_steps % self.opt.plot_losses_freq == 0 and self.visual.display_id > 0:
                cur_losses = self.train_model.get_latest_losses()
                epoch_steps = idx * self.opt.batch_size
                self.visual.display_current_losses(epoch - 1, epoch_steps / len(self.train_dataset), cur_losses)

            if self.train_total_steps % self.opt.sample_img_freq == 0 and self.visual.display_id > 0:
                cur_vis = self.train_model.get_latest_visuals()
                self.visual.display_online_results(cur_vis, epoch)

    def test_networks(self, opt):
        self.init_test_setting(opt)
        self.test_ops()

    def init_test_setting(self, opt):
        self.test_dataset = create_dataloader(opt)
        self.test_model = create_model(opt)

    def test_ops(self):
        for batch_idx, batch in enumerate(self.test_dataset):
            with torch.no_grad():
                faces_list = [batch['src_img'].float().numpy()]
                paths_list = [batch['src_path'], batch['tar_path']]
                for idx in range(self.opt.interpolate_len):
                    cur_alpha = (idx + 1.) / float(self.opt.interpolate_len)
                    cur_tar_aus = cur_alpha * batch['tar_aus'] + (1 - cur_alpha) * batch['src_aus']
                    test_batch = {'src_img': batch['src_img'], 'tar_aus': cur_tar_aus, 'src_aus': batch['src_aus'],
                                  'tar_img': batch['tar_img']}

                    self.test_model.feed_batch(test_batch)
                    self.test_model.forward()

                    cur_gen_faces = self.test_model.fake_img.cpu().float().numpy()
                    faces_list.append(cur_gen_faces)
                faces_list.append(batch['tar_img'].float().numpy())
            self.test_save_imgs(faces_list, paths_list)

    def test_save_imgs(self, faces_list, paths_list):
        for idx in range(len(paths_list[0])):
            src_name = os.path.splitext(os.path.basename(paths_list[0][idx]))[0]
            tar_name = os.path.splitext(os.path.basename(paths_list[1][idx]))[0]

            if self.opt.save_test_gif:
                imgs_numpy_list = []
                for face_idx in range(len(faces_list) - 1):
                    cur_numpy = np.array(self.visual.numpy2im(faces_list[face_idx][idx]))
                    imgs_numpy_list.extend([cur_numpy for _ in range
