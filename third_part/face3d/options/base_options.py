import argparse
import os
from util import util
import torch
import face3d.models as models
import face3d.data as data

class BaseOptions():
    def __init__(self, cmd_line=None):
        # Check if command line arguments are provided
        self.initialized = False
        self.cmd_line = cmd_line.split() if cmd_line else None

    def initialize(self, parser):
        # Add common options for both training and test
        parser.add_argument('--name', type=str, default='face_recon', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # ... (other options)
        self.initialized = True
        return parser

    def gather_options(self):
        # Check if the class has been initialized
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        # Parse command line arguments
        opt, _ = parser.parse_known_args(self.cmd_line)
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args(self.cmd_line)
        # Check if dataset_mode is specified
        if opt.dataset_mode:
            dataset_name = opt.dataset_mode
            dataset_option_setter = data.get_option_setter(dataset_name)
            parser = dataset_option_setter(parser, self.isTrain)
        self.parser = parser
        # Return parsed options
        return parser.parse_args() if self.cmd_line is None else parser.parse_args(self.cmd_line)

    def print_options(self, opt):
        # Print options for better visibility
        message = '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        # Save options to a text file
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        try:
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
        except PermissionError as error:
            # Handle permission error
            print("Permission error: {}".format(error))
            pass

    def parse(self):
        # Parse options
        opt = self.gather_options()
        opt.isTrain = self.isTrain
        # Process suffix and set name accordingly
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix
        str_ids = opt.gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                gpu_ids.append(id)
        opt.world_size = len(gpu_ids)
        if opt.world_size == 1:
            opt.use_ddp = False
        if opt.phase != 'test':
            # Check if continue_train should be set automatically
            if opt.pretrained_name is None:
                model_dir = os.path.join(opt.checkpoints_dir, opt.name)
            else:
                model_dir = os.path.join(opt.checkpoints_dir, opt.pretrained_name)
            if os.path.isdir(model_dir):
                model_pths = [i for i in os.listdir(model_dir) if i.endswith('pth')]
                if os.path.isdir(model_dir) and len(model_pths) != 0:
                    opt.continue_train = True
            # Update the latest epoch count
            if opt.continue_train:
                if opt.epoch == 'latest':
                    epoch_counts = [int(i.split('.')[0].split('_')[-1]) for i in model_pths if 'latest' not in i]
                    if len(epoch_counts) != 0:
                        opt.epoch_count = max(epoch_counts) + 1
                else:
                    opt.epoch_count = int(opt.epoch) + 1
        self.print_options(opt)
        self.opt = opt
        # Return parsed options
        return self.opt
