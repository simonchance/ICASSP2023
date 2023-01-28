import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from tqdm import tqdm
import numpy as np
import numpy
from model  import GBSNet
from my_dataset import CrowdDataset
import math
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from PIL import Image
from sewar.full_ref import ssim

def psnr2(img1, img2):
   #print(img1.shape)
   mse = ((img1 - img2) ** 2).sum() / (img1.shape[2] * img1.shape[3])
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = img2.max()
   #print(PIXEL_MAX)
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse.item()))


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        # pdb.set_trace()
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssimf(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def cal_mae(img_root, gt_dmap_root, model_param_path,index):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    device = torch.device("cuda")
    model = GBSNet()
    model.load_state_dict(torch.load(model_param_path, map_location='cuda:0'),strict=False)
    model.to(device)
    dataset = CrowdDataset(img_root, gt_dmap_root, 8, phase='test')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    mae = 0
    mse = 0
    psnr = 0
    ssimxmy = 0
    with torch.no_grad():
        for i, (img, gt_dmap) in enumerate(dataloader):
            #if i is not index:
                #continue
            img = img.to(device)

            gt_dmap = gt_dmap.to(device)
            # forward propagation
            et_dmap = model(img)


            mae += abs(et_dmap.data.sum() - gt_dmap.data.sum()).item()
            mse += np.square((et_dmap.data.sum() - gt_dmap.data.sum()).item())
            psnr += psnr2(et_dmap, gt_dmap)

            ett_dmap = np.array(et_dmap.cpu())
            gtt_dmap = np.array(gt_dmap.cpu())
            max1 = et_dmap.max()
            max2 = gt_dmap.max()
            max1 = np.array(max1.cpu())
            max2 = np.array(max2.cpu())

            ett_dmap = ett_dmap.astype(np.float64) / max1
            gtt_dmap = gtt_dmap.astype(np.float64) / max2

            ettt_dmap = 255 * ett_dmap
            gttt_dmap = 255 * gtt_dmap
            etttt_dmap = ettt_dmap.astype(np.uint8)
            gtttt_dmap = gttt_dmap.astype(np.uint8)
            etttt_dmap = np.squeeze(etttt_dmap)
            gtttt_dmap = np.squeeze(gtttt_dmap)
            ssimxmy += ssim(gtttt_dmap, etttt_dmap)[0]
            print(round(et_dmap.data.sum().item(), 2), round(gt_dmap.data.sum().item(), 2),
                  round(abs(et_dmap.data.sum() - gt_dmap.data.sum()).item(), 2), round(psnr2(et_dmap, gt_dmap), 2,),
                  ssim(gtttt_dmap, etttt_dmap)[0])
            # print('ssim:{:.4f}'.format(ssim(gt_dmap,et_dmap).data))
            et_count = round(et_dmap.data.sum().item(), 2)
            gt_count = round(gt_dmap.data.sum().item(), 2)
            per_mae = round(abs(et_dmap.data.sum() - gt_dmap.data.sum()).item(), 2)
            per_psnr = round(psnr2(et_dmap, gt_dmap), 2)
            per_ssimv = ssim(gtttt_dmap, etttt_dmap)[0]
            # if not os.path.exists('2.txt'):
            # os.makedirs('2.txt')
            f = open('2.txt', 'a')
            # f.write(str(i)+'  et_dmap:'+str(et_count)+'     gt_dmap:'+str(gt_count)+'     mae:'+str(per_mae)+'     0psnr:'+str(per_psnr)+'\n')
            f.write(str(et_count) + ',' + str(gt_count) + ',' + str(per_mae) + ',' + str(per_psnr) + ',' + str(
                per_ssimv) + '\n')
            f.close()
            del img, gt_dmap, et_dmap
        mse = np.sqrt(mse / len(dataloader))
        mae = mae / len(dataloader)
        psnr = psnr / len(dataloader)
        ssimave = ssimxmy / len(dataloader)
        print("min_mae:%.2f min_mse:%.2f min_psnr:%.2f min_ssim:%.2f" % (mae, mse, psnr, ssimave))

def estimate_density_map(img_root,gt_dmap_root,model_param_path,index):
    '''
    Show one estimated density-map.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    index: the order of the test image in test dataset.
    '''
    device=torch.device("cuda")
    model=GBSNet().to(device)
    model.load_state_dict(torch.load(model_param_path,map_location='cuda:0'),strict=False)
    dataset=CrowdDataset(img_root,gt_dmap_root,8,phase='test')
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    model.eval()
    ssimvv = 0
    with torch.no_grad():
        for i,(img,gt_dmap) in enumerate(dataloader):
                #if i is not index:
                    #continue
                img1=img.squeeze(0).squeeze(0).numpy()
                #print(img.)
                img1=img1.transpose((1,2,0)) # convert to order (channel,rows,cols)   h,w,c
                print(img1.shape)

                
                imgp = img1 / img1.max()
                imgp = imgp*255
                imgp = np.clip(imgp,0,255)
                imgp = imgp.astype(np.uint8)
                imgpp = cv2.cvtColor(imgp, cv2.COLOR_RGB2BGR)
                #cv2.imshow('ogshow',imgpp)
                #cv2.waitKey(0)
                #cv2.destroyWindow('ogshow')
                cv2.imwrite('./og64.png', imgp)
                
                
                
                img=img.to(device)
                #forward propagation
                et_dmap=model(img).detach()
                ee = et_dmap.data.sum()
                maet = abs(ee-gt_dmap.data.sum()).item()
                print(maet)

                et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()
                gt_dmap=gt_dmap.squeeze(0).squeeze(0).cpu().numpy()

                #print(et_dmap.sum())
                #print(gt_dmap.sum())
                #print(et_dmap.sum() - gt_dmap.sum())
                
                et_dmap = et_dmap / et_dmap.max()
                gt_dmap = gt_dmap / gt_dmap.max()
                et_dmap = et_dmap*255
                gt_dmap = gt_dmap*255
                et_dmap = np.clip(et_dmap,0,255)
                gt_dmap = np.clip(gt_dmap,0,255)

                et_dmap = et_dmap.astype(np.uint8)
                gt_dmap = gt_dmap.astype(np.uint8)
                


                heat_et = cv2.applyColorMap(et_dmap,cv2.COLORMAP_JET)

                cv2.imwrite('./heat_et64.png',heat_et)


                heat_gt = cv2.applyColorMap(gt_dmap, cv2.COLORMAP_JET)
                #cv2.imshow('gtshow', heat_gt)
                #cv2.waitKey(0)
                #cv2.destroyWindow('gtshow')
                cv2.imwrite('./heat_gt64.png', heat_gt)
                


if __name__=="__main__":
    torch.backends.cudnn.enabled=False
    img_root='/home/dpd/Crowd_Counting/CSRNet-master/ShanghaiTech_Dataset/part_A_final/test_data/imagess'
    gt_dmap_root='/home/dpd/Crowd_Counting/CSRNet-master/ShanghaiTech_Dataset/part_A_final/test_data/ground_truths'

    model_param_path = './checkpoints/saved_epoch.pth'
    torch.cuda.set_device(0)
    print("using device:",0)
    #cal_mae(img_root,gt_dmap_root,model_param_path)
    estimate_density_map(img_root,gt_dmap_root,model_param_path,32)
    cal_mae(img_root, gt_dmap_root, model_param_path,32)
