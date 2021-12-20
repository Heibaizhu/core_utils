from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from .psnr import PSNR as TENSOR_PSNR
from .ssim import SSIM as TENSOR_SSIM
from .ciede2000 import rgb2lab, ciede2000
# from .ciede2000_ori import rgb2lab, ciede2000
import pdb

EPSILON = 1e-10

class AverageMeter:
    def __init__(self, callback=None):
        super().__init__()
        if callback is not None:
            self.compute = callback
        self.reset()

    def compute(self, *args):
        if len(args) == 1:
            return args[0]
        else:
            raise NotImplementedError

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, *args, n=1):
        self.val = self.compute(*args)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count

    def __call__(self, *args):
        return self.compute(*args)

class PSNR(AverageMeter):
    __name__ = 'PSNR'
    def __init__(self, **configs):
        super().__init__(callback=partial(psnr, **configs))

class PSNR_TENSOR(AverageMeter):
    __name__ = 'PSNR_TENSOR'
    def __init__(self, **configs):
        configs.setdefault('data_range', 1)
        psnr = TENSOR_PSNR(**configs)
        super().__init__(callback=psnr)


class SSIM(AverageMeter):
    __name__ = 'SSIM'
    def __init__(self, **configs):
        super().__init__(callback=partial(ssim, **configs))

class SSIM_TENSOR(AverageMeter):
    __name__ = 'SSIM_TENSOR'
    def __init__(self, **configs):
        configs.setdefault('data_range', 1)
        ssim = TENSOR_SSIM(**configs)
        super().__init__(callback=ssim)

class RMSE(AverageMeter):
    __name__ = 'RMSE'
    def __init__(self):
        super().__init__(None)
    def compute(self, x1, x2):
        x1, x2 = x1.astype('float'), x2.astype('float')
        return np.sqrt(np.mean((x1-x2)**2))

class RMSE_TENSOR(AverageMeter):
    __name__ = 'RMSE_TENSOR'
    def __init__(self):
        super().__init__(None)
    def compute(self, x1, x2):
        # assert x1.shape != 4, 'Input images must 4-d tensor.'
        # assert x1.type() == x2.type(), 'Input images must 4-d tensor.'
        assert x1.shape == x2.shape, 'Input images must have the same dimensions.'
        x1, x2 = x1.type(torch.float), x2.type(torch.float)
        B = x1.shape[0]

        return torch.mean(torch.sqrt(torch.mean((x1.view(B, -1) - x2.view(B, -1))**2, dim=[1])))

class RELATIVE_ERROR(AverageMeter):
    """
    计算相对误差  abs(pred-gt)/(gt + gamma)
    """
    __name__ = 'RELATIVE_ERROR'
    def __init__(self, gamma=1):
        super().__init__(None)
        self.gamma = gamma

    def compute(self, pred, gt):
        assert pred.shape == gt.shape
        pred, gt = pred.float(), gt.float()
        relative_error = torch.div(torch.abs(pred - gt), gt + self.gamma)
        return torch.mean(relative_error)

class ABS_REL(AverageMeter):
    """
    absolute relative error
    """
    __name__ = 'ABS_REL'
    def __init__(self):
        super().__init__(None)
    def compute(self, pred, gt):
        pred, gt = pred.float(), gt.float()
        abs_rel = torch.abs(pred-gt) / (torch.abs(gt) + 1e-6)
        return torch.mean(abs_rel)

class MAE(AverageMeter):
    __name__ = 'MAE'
    def __init__(self):
        super().__init__(None)
    def compute(self, pred, gt):
        pred, gt = pred.float(), gt.float()
        mae = torch.abs(pred-gt)
        return torch.mean(mae)

class mae(AverageMeter):
    __name__ = 'mae'
    def __init__(self):
        super().__init__(None)
    def compute(self, pred, gt):
        pred, gt = pred.astype(np.float), gt.astype(np.float)
        mae = np.abs(pred-gt)
        return np.mean(mae)

class mse(AverageMeter):
    __name__ = 'mse'
    def __init__(self):
        super().__init__(None)
    def compute(self, pred, gt):
        pred, gt = pred.astype(np.float), gt.astype(np.float)
        mse = (pred-gt) ** 2
        return np.mean(mse)


class CCL(AverageMeter):
    __name__ = 'CCL'
    def __init__(self, win_size=11):
        super().__init__(None)
        self.win_size = win_size
    def compute(self, pred, gt):
        pred = np.expand_dims(pred, axis=0).transpose((0, 3, 1, 2))
        gt = np.expand_dims(gt, axis=0).transpose((0, 3, 1, 2))
        data = torch.from_numpy(pred).float()
        label = torch.from_numpy(gt).float()
        error = data - label 
        error_square = (data - label) ** 2
        B, C, H, W = data.shape 

        conv_weight = torch.ones((C, 1, self.win_size, self.win_size))
        N = self.win_size * self.win_size 
        pairs = N * (N - 1) / 2
        # pdb.set_trace()

        sum1 = F.conv2d(error_square, weight=conv_weight, groups=C)
        sum2 = F.conv2d(error, weight=conv_weight, groups=C)
        l = N * sum1 - sum2 ** 2 
        l = l.mean() / pairs
        return l.item()



class LOG_MAE(AverageMeter):
    __name__ = 'LOG_MAE'
    def __init__(self):
        super().__init__(None)
    def compute(self, pred, gt):
        pred, gt = pred.float(), gt.float()
        pred[pred <= 0] = 1e-9
        gt[gt <= 0] = 1e-9
        log_mae = torch.mean(torch.abs(torch.log(pred/gt)))
        return log_mae

class SQ_REL(AverageMeter):
    __name__ = 'SQ_REL'
    def __init__(self):
        super().__init__(None)
    def compute(self, pred, gt):
        pred, gt = pred.float(), gt.float()
        gt[gt <= 0] = 1e-9
        sq_rel = torch.mean(((pred-gt)/gt)**2)
        return sq_rel

class sq_rel(AverageMeter):
    __name__ = 'sq_rel'
    def __init__(self):
        super().__init__(None)
    def compute(self, pred, gt):
        pred, gt = pred.astype(np.float), gt.astype(np.float)
        gt[gt <= 0] = 1e-9
        sq_rel = np.mean(((pred-gt)/gt)**2)
        return sq_rel

class CIEDE2000(AverageMeter):
    __name__ = 'CIEDE2000'
    def __init__(self):
        super().__init__(None)
    def compute(self, pred, gt):
        pred, gt = pred.astype(np.float), gt.astype(np.float)
        #convert rgb to lab format
        pred_lab, gt_lab = rgb2lab(pred), rgb2lab(gt)
        result = ciede2000(pred_lab, gt_lab)

        # temp = np.zeros_like(pred[..., 0])
        # pred, gt = pred.reshape(-1, 3), gt.reshape(-1, 3)
        # result = np.zeros_like(pred[..., 0])
        # for i in range(pred.shape[0]):
        #     pred_rgb = pred[i, :]
        #     gt_rgb = gt[i, :]
        #     pred_lab, gt_lab = rgb2lab(pred_rgb), rgb2lab(gt_rgb)
        #     result[i] = ciede2000(pred_lab, gt_lab)


        # pdb.set_trace()
        return np.mean(result)

class CIEDE2000_TENSOR(AverageMeter):
    __name__ = 'CIEDE2000_TENSOR'
    def __init__(self):
        super().__init__(None)
    def compute(self, pred, gt):
        if isinstance(pred, torch.Tensor):
            B, C, H, W = pred.shape
            assert B == 1, "The batchsize of the pred is larger than 1!"
            pred = pred.squeeze(0).permute(1, 2, 0).to('cpu').numpy()
        if isinstance(gt, torch.Tensor):
            B, C, H, W = gt.shape
            assert B == 1, "The batchsize of the gt is larger than1!"
            gt = gt.squeeze(0).permute(1, 2, 0).to('cpu').numpy()
        if np.max(gt) <= 1:
            pred = pred * 255
            gt = gt * 255
        pred, gt = pred.astype(np.float), gt.astype(np.float)
        #convert rgb to lab format
        pred_lab, gt_lab = rgb2lab(pred), rgb2lab(gt)
        result = ciede2000(pred_lab, gt_lab)

        # temp = np.zeros_like(pred[..., 0])
        # pred, gt = pred.reshape(-1, 3), gt.reshape(-1, 3)
        # result = np.zeros_like(pred[..., 0])
        # for i in range(pred.shape[0]):
        #     pred_rgb = pred[i, :]
        #     gt_rgb = gt[i, :]
        #     pred_lab, gt_lab = rgb2lab(pred_rgb), rgb2lab(gt_rgb)
        #     result[i] = ciede2000(pred_lab, gt_lab)


        # pdb.set_trace()
        return np.mean(result)


if __name__ =='__main__':
   pass









