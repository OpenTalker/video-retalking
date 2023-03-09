import os
import glob
import numpy as np
from os import makedirs, name
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn

from face3d.options.inference_options import InferenceOptions
from face3d.models import create_model
from face3d.util.preprocess import align_img
from face3d.util.load_mats import load_lm3d
from face3d.extract_kp_videos import KeypointExtractor


class CoeffDetector(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.device = 'cuda'
        self.model.parallelize()
        self.model.eval()

        self.lm3d_std = load_lm3d(opt.bfm_folder) 

    def forward(self, img, lm):
        
        img, trans_params = self.image_transform(img, lm)

        data_input = {                
                'imgs': img[None],
                }        
        self.model.set_input(data_input)  
        self.model.test()
        pred_coeff = {key:self.model.pred_coeffs_dict[key].cpu().numpy() for key in self.model.pred_coeffs_dict}
        pred_coeff = np.concatenate([
            pred_coeff['id'], 
            pred_coeff['exp'], 
            pred_coeff['tex'], 
            pred_coeff['angle'],
            pred_coeff['gamma'],
            pred_coeff['trans'],
            trans_params[None],
            ], 1)
        
        return {'coeff_3dmm':pred_coeff, 
                'crop_img': Image.fromarray((img.cpu().permute(1, 2, 0).numpy()*255).astype(np.uint8))}

    def image_transform(self, images, lm):
        """
        param:
            images:          -- PIL image 
            lm:              -- numpy array
        """
        W,H = images.size
        if np.mean(lm) == -1:
            lm = (self.lm3d_std[:, :2]+1)/2.
            lm = np.concatenate(
                [lm[:, :1]*W, lm[:, 1:2]*H], 1
            )
        else:
            lm[:, -1] = H - 1 - lm[:, -1]

        trans_params, img, lm, _ = align_img(images, lm, self.lm3d_std)        
        img = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2, 0, 1)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)])
        trans_params = torch.tensor(trans_params.astype(np.float32))
        return img, trans_params        

def get_data_path(root, keypoint_root):
    filenames = list()
    keypoint_filenames = list()

    IMAGE_EXTENSIONS_LOWERCASE = {'jpg', 'png', 'jpeg', 'webp'}
    IMAGE_EXTENSIONS = IMAGE_EXTENSIONS_LOWERCASE.union({f.upper() for f in IMAGE_EXTENSIONS_LOWERCASE})
    extensions = IMAGE_EXTENSIONS

    for ext in extensions:
        filenames += glob.glob(f'{root}/*.{ext}', recursive=True)
    filenames = sorted(filenames)
    for filename in filenames:
        name = os.path.splitext(os.path.basename(filename))[0]
        keypoint_filenames.append(
            os.path.join(keypoint_root, name + '.txt')
        )
    return filenames, keypoint_filenames


if __name__ == "__main__":
    opt = InferenceOptions().parse() 
    coeff_detector = CoeffDetector(opt)
    kp_extractor = KeypointExtractor()
    image_names, keypoint_names = get_data_path(opt.input_dir, opt.keypoint_dir)
    makedirs(opt.keypoint_dir, exist_ok=True)
    makedirs(opt.output_dir, exist_ok=True)

    for image_name, keypoint_name in tqdm(zip(image_names, keypoint_names)):
        image = Image.open(image_name)
        if not os.path.isfile(keypoint_name):
            lm = kp_extractor.extract_keypoint(image, keypoint_name)
        else:
            lm = np.loadtxt(keypoint_name).astype(np.float32)
            lm = lm.reshape([-1, 2]) 
        predicted = coeff_detector(image, lm)
        name = os.path.splitext(os.path.basename(image_name))[0]
        np.savetxt(
            "{}/{}_3dmm_coeff.txt".format(opt.output_dir, name), 
            predicted['coeff_3dmm'].reshape(-1))

        



    