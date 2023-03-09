import os
import cv2
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.io import savemat

import torch 

from models import create_model
from options.inference_options import InferenceOptions
from util.preprocess import align_img
from util.load_mats import load_lm3d
from util.util import mkdirs, tensor2im, save_image


def get_data_path(root, keypoint_root):
    filenames = list()
    keypoint_filenames = list()

    VIDEO_EXTENSIONS_LOWERCASE = {'mp4'}
    VIDEO_EXTENSIONS = VIDEO_EXTENSIONS_LOWERCASE.union({f.upper() for f in VIDEO_EXTENSIONS_LOWERCASE})
    extensions = VIDEO_EXTENSIONS

    for ext in extensions:
        filenames += glob.glob(f'{root}/**/*.{ext}', recursive=True)
    filenames = sorted(filenames)
    keypoint_filenames = sorted(glob.glob(f'{keypoint_root}/**/*.txt', recursive=True))
    assert len(filenames) == len(keypoint_filenames)

    return filenames, keypoint_filenames

class VideoPathDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, txt_filenames, bfm_folder):
        self.filenames = filenames
        self.txt_filenames = txt_filenames
        self.lm3d_std = load_lm3d(bfm_folder) 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        txt_filename = self.txt_filenames[index]
        frames = self.read_video(filename)
        lm = np.loadtxt(txt_filename).astype(np.float32)
        lm = lm.reshape([len(frames), -1, 2]) 
        out_images, out_trans_params = list(), list()
        for i in range(len(frames)):
            out_img, _, out_trans_param \
                = self.image_transform(frames[i], lm[i])
            out_images.append(out_img[None])
            out_trans_params.append(out_trans_param[None])
        return {
            'imgs': torch.cat(out_images, 0),
            'trans_param':torch.cat(out_trans_params, 0),
            'filename': filename
        }
        
    def read_video(self, filename):
        frames = list()
        cap = cv2.VideoCapture(filename)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(frame)
            else:
                break
        cap.release()
        return frames

    def image_transform(self, images, lm):
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
        lm = torch.tensor(lm)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)])
        trans_params = torch.tensor(trans_params.astype(np.float32))
        return img, lm, trans_params        

def main(opt, model):
    # import torch.multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')
    filenames, keypoint_filenames = get_data_path(opt.input_dir, opt.keypoint_dir)
    dataset = VideoPathDataset(filenames, keypoint_filenames, opt.bfm_folder)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, # can noly set to one here!
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )     
    batch_size = opt.inference_batch_size
    for data in tqdm(dataloader):
        num_batch = data['imgs'][0].shape[0] // batch_size + 1
        pred_coeffs = list()
        for index in range(num_batch):
            data_input = {                
                'imgs': data['imgs'][0,index*batch_size:(index+1)*batch_size],
            }
            model.set_input(data_input)  
            model.test()
            pred_coeff = {key:model.pred_coeffs_dict[key].cpu().numpy() for key in model.pred_coeffs_dict}
            pred_coeff = np.concatenate([
                pred_coeff['id'], 
                pred_coeff['exp'], 
                pred_coeff['tex'], 
                pred_coeff['angle'],
                pred_coeff['gamma'],
                pred_coeff['trans']], 1)
            pred_coeffs.append(pred_coeff) 
            visuals = model.get_current_visuals()  # get image results
            if False: # debug
                for name in visuals:
                    images = visuals[name]
                    for i in range(images.shape[0]):
                        image_numpy = tensor2im(images[i])
                        save_image(
                            image_numpy, 
                            os.path.join(
                                opt.output_dir,
                                os.path.basename(data['filename'][0])+str(i).zfill(5)+'.jpg')
                            )
                exit()

        pred_coeffs = np.concatenate(pred_coeffs, 0)
        pred_trans_params = data['trans_param'][0].cpu().numpy()
        name = data['filename'][0].split('/')[-2:]
        name[-1] = os.path.splitext(name[-1])[0] + '.mat'
        os.makedirs(os.path.join(opt.output_dir, name[-2]), exist_ok=True)
        savemat(
            os.path.join(opt.output_dir, name[-2], name[-1]), 
            {'coeff':pred_coeffs, 'transform_params':pred_trans_params}
        )

if __name__ == '__main__':
    opt = InferenceOptions().parse()  # get test options
    model = create_model(opt)
    model.setup(opt)
    model.device = 'cuda:0'
    model.parallelize()
    model.eval()

    main(opt, model)


