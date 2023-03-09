from .base_dataset import BaseDataset
import os
import random
import numpy as np


class CelebADataset(BaseDataset):
    """docstring for CelebADataset"""
    def __init__(self):
        super(CelebADataset, self).__init__()
        
    def initialize(self, opt):
        super(CelebADataset, self).initialize(opt)

    def get_aus_by_path(self, img_path):
        assert os.path.isfile(img_path), "Cannot find image file: %s" % img_path
        img_id = str(os.path.splitext(os.path.basename(img_path))[0])
        return self.aus_dict[img_id] / 5.0   # norm to [0, 1]

    def make_dataset(self):
        # return all image full path in a list
        imgs_path = []
        assert os.path.isfile(self.imgs_name_file), "%s does not exist." % self.imgs_name_file
        with open(self.imgs_name_file, 'r') as f:
            lines = f.readlines()
            imgs_path = [os.path.join(self.imgs_dir, line.strip()) for line in lines]
            imgs_path = sorted(imgs_path)
        return imgs_path

    def __getitem__(self, index):
        img_path = self.imgs_path[index]

        # load source image
        src_img = self.get_img_by_path(img_path)
        src_img_tensor = self.img2tensor(src_img)
        src_aus = self.get_aus_by_path(img_path)

        # load target image
        tar_img_path = random.choice(self.imgs_path)
        tar_img = self.get_img_by_path(tar_img_path)
        tar_img_tensor = self.img2tensor(tar_img)
        tar_aus = self.get_aus_by_path(tar_img_path)
        if self.is_train and not self.opt.no_aus_noise:
            tar_aus = tar_aus + np.random.uniform(-0.1, 0.1, tar_aus.shape)

        # record paths for debug and test usage
        data_dict = {'src_img':src_img_tensor, 'src_aus':src_aus, 'tar_img':tar_img_tensor, 'tar_aus':tar_aus, \
                        'src_path':img_path, 'tar_path':tar_img_path}

        return data_dict
