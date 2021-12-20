# -*- coding: utf-8 -*-

import cv2
import math
from PIL import Image
import numpy as np
import os

import torch
from torchvision.utils import make_grid


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
            img = torch.from_numpy(img)
        else:
            if img.shape[2] == 3 and bgr2rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(-1, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [-1, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))),
                normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. '
                            f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED
    }
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def yuv2bgr(content, flag=cv2.COLOR_YUV2BGR_I420):
    """Convert an image from YUV space to BGR space.

    Args:
        content (ndarray): A YUV image
        flag (str): Flags specifying the type of color conversion.

    Returns:
        img (ndarray): A BGR image
    """
    img = cv2.cvtColor(content, flag)
    return img


def bgr2yuv(content, flag=cv2.COLOR_BGR2YUV_I420):
    """Convert an image from BGR space to YUV space.

    Args:
        content (ndarray): A BGR image
        flag (str): Flags specifying the type of color conversion.

    Returns:
        img (ndarray): A YUV image
    """
    img = cv2.cvtColor(content, flag)
    return img


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


def crop_border(imgs, crop_border):
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [
                v[crop_border:-crop_border, crop_border:-crop_border, ...]
                for v in imgs
            ]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border,
                        ...]


def img_resize(img, dsize=None, scale=None, type=cv2.INTER_CUBIC):
    if dsize is None and scale is None:
        raise AssertionError('dsize and scale are None')
    elif(dsize is not None and scale is not None):
        raise AssertionError('Only support dsize or scale to resize image. '
                             'But received both dsize and scale.')
    elif dsize is None:
        H, W = img.shape[:2]
        dsize = (int(W*scale), int(H*scale))
    out_img = cv2.resize(img, dsize, interpolation=type)
    return out_img


def yuv_pil_resize(lr_y, lr_u=None, lr_v=None, scale=2):
    y_w = lr_y.shape[1]
    y_h = lr_y.shape[0]
    bic_y = Image.fromarray(lr_y).resize((y_w * scale, y_h * scale), Image.BICUBIC)
    bic_y = np.asarray(bic_y)

    if lr_u is not None and lr_v is not None:
        uv_h = lr_u.shape[0]
        uv_w = lr_u.shape[1]
        bic_u = Image.fromarray(lr_u).resize((uv_w * scale, uv_h * scale), Image.BICUBIC)
        bic_u = np.asarray(bic_u)
        bic_v = Image.fromarray(lr_v).resize((uv_w * scale, uv_h * scale), Image.BICUBIC)
        bic_v = np.asarray(bic_v)
        return bic_y, bic_u, bic_v
    else:
        return bic_y, None, None


def depth2flow(z_map, ori_parm, tar_parm):
    H, W = z_map.shape[:2]
    K_ori, R_ori, t_ori = ori_parm['K'], ori_parm['R'], ori_parm['t']
    K_tar, R_tar, t_tar = tar_parm['K'], tar_parm['R'], tar_parm['t']
    fx, fy = K_ori[0, 0], K_ori[1, 1]
    u0, v0 = K_ori[0, 2], K_ori[1, 2]
    P_tar = K_tar@np.concatenate([R_tar, t_tar], axis=1)
    x_map_ori, y_map_ori = np.meshgrid(np.arange(W), np.arange(H))
    x_map_ori, y_map_ori = x_map_ori.astype(np.float32), y_map_ori.astype(np.float32)
    # project image coordinates to source camera coordinates
    x_map = (x_map_ori-u0)*z_map/fx
    y_map = (y_map_ori-v0)*z_map/fy
    xyz_map = np.concatenate((x_map.reshape(1, H*W),
                                y_map.reshape(1, H*W),
                                z_map.reshape(1, H*W)), axis=0)
    # project camera coordinates to world coordinates
    xyz_map = xyz_map - t_ori
    xyz_map = R_ori.I@xyz_map
    xyz_map = np.concatenate((xyz_map, np.ones((1, H*W))), axis=0)
    # project world coordinates to target image coordinates
    xyz_map = P_tar@xyz_map
    # project x,y to homogeneous coordinates
    xyz_map[0, :] = xyz_map[0, :]/(xyz_map[2, :]+1e-8)
    xyz_map[1, :] = xyz_map[1, :]/(xyz_map[2, :]+1e-8)
    # back warp mapping
    xy_flow = np.asarray(xyz_map[:2, :]).reshape(2, H, W)
    xy_flow[0, :] = (xy_flow[0, :] - x_map_ori)
    xy_flow[1, :] = (xy_flow[1, :] - y_map_ori)
    # xy_flow[0, :] = xy_flow[0, :] / W *2-1
    # xy_flow[1, :] = xy_flow[1, :] /H*2-1
    return xy_flow
