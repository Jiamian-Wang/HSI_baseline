import os
import time
import torch
import datetime
import numpy as np
from utils import *
import models_resblock_v2
import models_resblock_v3
import models_resblock_v4

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if not torch.cuda.is_available():
    raise Exception('NO GPU!')
else:
    print('CUDA is available')



mask_path = "/data/jiamianw/ICCV_arXiv/Data"
test_path = "/data/jiamianw/ICCV_arXiv/Data/testing/simu/"
batch_size = 1
patch_size = 256
logger = None

mask3d_batch = generate_masks(mask_path, batch_size)
test_data = LoadTest(test_path, patch_size)
model_path = '/data/jiamianw/ICCV_arXiv/models/v1/model_epoch_204.pth'

model = torch.load(model_path)

def test(logger):
    psnr_list, ssim_list = [], []
    test_gt = test_data.cuda().float()
    test_PhiTy = gen_meas_torch(test_gt, mask3d_batch, is_training = False)
    begin = time.time()
    with torch.no_grad():
        model_out = model(test_PhiTy)
    end = time.time()
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k,:,:,:], test_gt[k,:,:,:])
        ssim_val = torch_ssim(model_out[k,:,:,:], test_gt[k,:,:,:])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
        print('psnr=', psnr_val, 'ssim=', ssim_val)
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    print('===> testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'.format(psnr_mean, ssim_mean, (end - begin)))
    return (pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean)
    
     
def main():
    (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(logger)


if __name__ == '__main__':
    main()    
    

