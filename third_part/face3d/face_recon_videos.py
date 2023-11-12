import os, cv2, glob, numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.io import savemat
import torch
from models import create_model
from options.inference_options import InferenceOptions
from util.preprocess import align_img
from util.load_mats import load_lm3d
from util.util import tensor2im, save_image
from util import util as utility

def get_data_path(root, keypoint_root):
    filenames = sorted(glob.glob(f'{root}/**/*.mp4', recursive=True))
    keypoint_filenames = sorted(glob.glob(f'{keypoint_root}/**/*.txt', recursive=True))
    assert len(filenames) == len(keypoint_filenames)
    return filenames, keypoint_filenames  

class VideoPathDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, txt_filenames, bfm_folder):
        self.filenames = sorted(filenames)
        self.txt_filenames = sorted(txt_filenames)
        self.lm3d_std = load_lm3d(bfm_folder) 

    def __getitem__(self, index):
        filename = self.filenames[index]
        txt_filename = self.txt_filenames[index]
        frames = self.read_video(filename)
        lm = np.loadtxt(txt_filename).astype(np.float32)
        lm = lm.reshape([len(frames), -1, 2]) 
        out_images, out_trans_params = self.process_frames(frames, lm)
        return {
            'imgs': torch.cat(out_images, 0),
            'trans_param': torch.cat(out_trans_params, 0),
            'filename': filename
        }

    def process_frames(self, frames, lm):
        out_images, out_trans_params = [], []
        for i in range(len(frames)):
            out_img, _, out_trans_param = self.image_transform(frames[i], lm[i])
            out_images.append(out_img[None])
            out_trans_params.append(out_trans_param[None])
        return out_images, out_trans_params
        
    def read_video(self, filename):
        frames = []
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
        W, H = images.size
        if np.mean(lm) == 0:
            lm = (self.lm3d_std[:, :2] + 1) / 2.
            lm = np.concatenate([lm[:, :1] * W, lm[:, 1:2] * H], 1)
        else:
            lm[:, -1] = H - 1 - lm[:, -1]

        trans_params, img, lm, _ = align_img(images, lm, self.lm3d_std)
        img = torch.tensor(np.array(img) / 255., dtype=torch.float32).permute(2, 0, 1)
        lm = torch.tensor(lm)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)])
        trans_params = torch.tensor(trans_params.astype(np.float32))
        return img, lm, trans_params

def main(opt, model):
    filenames, keypoint_filenames = get_data_path(opt.input_dir, opt.keypoint_dir)
    dataset = VideoPathDataset(filenames, keypoint_filenames, opt.bfm_folder)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )     
    batch_size = opt.inference_batch_size
    for data in tqdm(dataloader):
        num_batch = data['imgs'][0].shape[0]
        pred_coeffs = self.process_batches(data, batch_size, model)
        visuals = model.get_current_visuals()
        self.debug_visuals(visuals, data, opt)
        self.save_output_data(opt, data, pred_coeffs)

def process_batches(self, data, batch_size, model):
    pred_coeffs = []
    for index in range(num_batch):
        data_input = {                
            'imgs': data['imgs'][0, index * batch_size:(index + 1) * batch_size],
        }
        model.set_input(data_input)  
        model.test()
        pred_coeff = {key: model.pred_coeffs_dict[key].cpu().numpy() for key in model.pred_coeffs_dict}
        pred_coeff = np.concatenate([
            pred_coeff['id'], 
            pred_coeff['exp'], 
            pred_coeff['tex'], 
            pred_coeff['angle'],
            pred_coeff['gamma'],
            pred_coeff['trans']], 1)
        pred_coeffs.append(pred_coeff) 
    pred_coeffs = np.concatenate(pred_coeffs, 0)
    return pred_coeffs

def debug_visuals(self, visuals, data, opt):
    for name in visuals:
        images = visuals[name]
        for i in range(images.shape[0]):
            image_numpy = tensor2im(images[i])
            save_image(
                image_numpy, 
                os.path.join( opt.output_dir,
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

