import numpy as np
from utils.warplayer import warp
import torch

def get_mpi_flow(img_size, K_src, R_src, t_src, K_tgt, R_tgt, t_tgt, depth_list):
    img_h, img_w = img_size
    flow_list = []
    for depth in depth_list:
        H = get_inv_homography(K_src, R_src, t_src, K_tgt, R_tgt, t_tgt, depth)
        flow = get_flow_by_homography(img_h, img_w, H)
        flow_list.append(flow[np.newaxis])
    return np.concatenate(flow_list, axis=0)


def get_xyz_from_depth(meshgrid_homo,
                       depth,
                       K_inv):
    """

    :param meshgrid_homo: 3xHxW
    :param depth: Bx1xHxW
    :param K_inv: Bx3x3
    :return: xyz: [B, 3, H, W]
    """
    H, W = meshgrid_homo.size(1), meshgrid_homo.size(2)
    B, _, H_d, W_d = depth.size()
    assert H==H_d, W==W_d

    # 3xHxW -> Bx3xHxW
    meshgrid_src_homo = meshgrid_homo.unsqueeze(0).repeat(B, 1, 1, 1)
    meshgrid_src_homo_B3N = meshgrid_src_homo.reshape(B, 3, -1)
    xyz_src = torch.matmul(K_inv, meshgrid_src_homo_B3N)  # Bx3xHW
    xyz_src = xyz_src.reshape(B, 3, H, W) * depth  # Bx3xHxW

    return xyz_src

def render(rgb_BS3HW, sigma_BS1HW, xyz_BS3HW, use_alpha=False, is_bg_depth_inf=False):
    if not use_alpha:
        imgs_syn, depth_syn, blend_weights, weights = plane_volume_rendering(
            rgb_BS3HW,
            sigma_BS1HW,
            xyz_BS3HW,
            is_bg_depth_inf
        )
    else:
        imgs_syn, weights = alpha_composition(sigma_BS1HW, rgb_BS3HW)
        depth_syn, _ = alpha_composition(sigma_BS1HW, xyz_BS3HW[:, :, 2:])
        # No rgb blending with alpha composition
        blend_weights = torch.zeros_like(rgb_BS3HW).cuda()
    return imgs_syn, depth_syn, blend_weights, weights


def alpha_composition(alpha_BK1HW, value_BKCHW):
    """
    composition equation from 'Single-View View Synthesis with Multiplane Images'
    K is the number of planes, k=0 means the nearest plane, k=K-1 means the farthest plane
    :param alpha_BK1HW: alpha at each of the K planes
    :param value_BKCHW: rgb/disparity at each of the K planes
    :return:
    """
    B, K, _, H, W = alpha_BK1HW.size()
    alpha_comp_cumprod = torch.cumprod(1 - alpha_BK1HW, dim=1)  # BxKx1xHxW

    preserve_ratio = torch.cat((torch.ones((B, 1, 1, H, W), dtype=alpha_BK1HW.dtype, device=alpha_BK1HW.device),
                                alpha_comp_cumprod[:, 0:K-1, :, :, :]), dim=1)  # BxKx1xHxW
    weights = alpha_BK1HW * preserve_ratio  # BxKx1xHxW
    value_composed = torch.sum(value_BKCHW * weights, dim=1, keepdim=False)  # Bx3xHxW

    return value_composed, weights


def get_src_xyz_from_plane_disparity(meshgrid_src_homo,
                                     mpi_disparity_src,
                                     K_src_inv):
    """

    :param meshgrid_src_homo: 3xHxW
    :param mpi_disparity_src: BxS
    :param K_src_inv: Bx3x3
    :return: xyz_src_BS3HW  BxSx3xHxW
    """
    device = meshgrid_src_homo.device
    mpi_disparity_src = mpi_disparity_src.to(device)
    K_src_inv = K_src_inv.to(device)
    B, S = mpi_disparity_src.size()
    H, W = meshgrid_src_homo.size(1), meshgrid_src_homo.size(2)
    mpi_depth_src = torch.reciprocal(mpi_disparity_src)  # BxS

    K_src_inv_Bs33 = K_src_inv.unsqueeze(1).repeat(1, S, 1, 1).reshape(B * S, 3, 3)

    # 3xHxW -> BxSx3xHxW
    meshgrid_src_homo = meshgrid_src_homo.unsqueeze(0).unsqueeze(1).repeat(B, S, 1, 1, 1)
    meshgrid_src_homo_Bs3N = meshgrid_src_homo.reshape(B * S, 3, -1)
    xyz_src = torch.matmul(K_src_inv_Bs33, meshgrid_src_homo_Bs3N)  # BSx3xHW
    # xyz_src = xyz_src / (xyz_src[:, 3:, ...] + 1e-18)
    # xyz_src = xyz_src[:, :3, ...]
    xyz_src = xyz_src.reshape(B, S, 3, H * W) * mpi_depth_src.unsqueeze(2).unsqueeze(3)  # BxSx3xHW
    xyz_src_BS3HW = xyz_src.reshape(B, S, 3, H, W)

    return xyz_src_BS3HW

def get_tgt_xyz_from_plane_disparity(xyz_src_BS3HW,
                                     G_tgt_src):
    """

    :param xyz_src_BS3HW: BxSx3xHxW
    :param G_tgt_src: Bx4x4
    :return:
    """

    def transform_G_xyz(G, xyz, is_return_homo=False):
        """

        :param G: Bx4x4
        :param xyz: Bx3xN
        :return:
        """
        assert len(G.size()) == len(xyz.size())
        if len(G.size()) == 2:
            G_B44 = G.unsqueeze(0)
            xyz_B3N = xyz.unsqueeze(0)
        else:
            G_B44 = G
            xyz_B3N = xyz
        xyz_B4N = torch.cat((xyz_B3N, torch.ones_like(xyz_B3N[:, 0:1, :])), dim=1)
        G_xyz_B4N = torch.matmul(G_B44, xyz_B4N)
        if is_return_homo:
            return G_xyz_B4N
        else:
            return G_xyz_B4N[:, 0:3, :]
    B, S, _, H, W = xyz_src_BS3HW.size()
    G_tgt_src_Bs33 = G_tgt_src.unsqueeze(1).repeat(1, S, 1, 1).reshape(B*S, 4, 4)
    xyz_tgt = transform_G_xyz(G_tgt_src_Bs33, xyz_src_BS3HW.reshape(B*S, 3, H*W))  # Bsx3xHW
    xyz_tgt_BS3HW = xyz_tgt.reshape(B, S, 3, H, W)  # BxSx3xHxW
    return xyz_tgt_BS3HW


def plane_volume_rendering(rgb_BS3HW, sigma_BS1HW, xyz_BS3HW, is_bg_depth_inf):
    """

    Args:
        rgb_BS3HW:
        sigma_BS1HW:
        xyz_BS3HW:  z from small to large
        is_bg_depth_inf:

    Returns:

    """
    def weighted_sum_mpi(rgb_BS3HW, xyz_BS3HW, weights, is_bg_depth_inf):
        weights_sum = torch.sum(weights, dim=1, keepdim=False)  # Bx1xHxW
        rgb_out = torch.sum(weights * rgb_BS3HW, dim=1, keepdim=False)  # Bx3xHxW

        if is_bg_depth_inf:
            depth_out = torch.sum(weights * xyz_BS3HW[:, :, 2:, :, :], dim=1, keepdim=False) \
                        + (1 - weights_sum) * 1000
        else:
            depth_out = torch.sum(weights * xyz_BS3HW[:, :, 2:, :, :], dim=1, keepdim=False) \
                        / (weights_sum + 1e-5)  # Bx1xHxW

        return rgb_out, depth_out

    B, S, _, H, W = sigma_BS1HW.size()

    xyz_diff_BS3HW = xyz_BS3HW[:, 1:, :, :, :] - xyz_BS3HW[:, 0:-1, :, :, :]  # Bx(S-1)x3xHxW
    xyz_dist_BS1HW = torch.norm(xyz_diff_BS3HW, dim=2, keepdim=True)  # Bx(S-1)x1xHxW

    xyz_dist_BS1HW = torch.cat((xyz_dist_BS1HW,
                                torch.full((B, 1, 1, H, W),
                                           fill_value=1e3,
                                           dtype=xyz_BS3HW.dtype,
                                           device=xyz_BS3HW.device)),
                               dim=1)  # BxSx3xHxW
    transparency = torch.exp(-sigma_BS1HW * xyz_dist_BS1HW)  # BxSx1xHxW
    alpha = 1 - transparency # BxSx1xHxW

    # add small eps to avoid zero transparency_acc
    # pytorch.cumprod is like: [a, b, c] -> [a, a*b, a*b*c], we need to modify it to [1, a, a*b]
    transparency_acc = torch.cumprod(transparency + 1e-6, dim=1)  # BxSx1xHxW
    transparency_acc = torch.cat((torch.ones((B, 1, 1, H, W), dtype=transparency.dtype, device=transparency.device),
                                  transparency_acc[:, 0:-1, :, :, :]),
                                 dim=1)  # BxSx1xHxW

    weights = transparency_acc * alpha  # BxSx1xHxW
    rgb_out, depth_out = weighted_sum_mpi(rgb_BS3HW, xyz_BS3HW, weights, is_bg_depth_inf)

    return rgb_out, depth_out, transparency_acc, weights


def get_G_target_src(src_parm, tar_parm):
    """

    Args:
        src_parm: src_parm['K']  3x3   src_parm['R'] 3x3   src_parm['T']  3x1
        tar_parm

    Returns:   G transform P_src to P_tgt     P_tgt = G_target_src x  P_src

    """
    # set source camera coordinates as the world coordinates
    R_s, t_s = src_parm['R'], src_parm['t']
    R_t, t_t = tar_parm['R'], tar_parm['t']
    Rt_s = np.concatenate([R_s, t_s], axis=1)
    Rt_s = np.concatenate([Rt_s, np.array([[0, 0, 0, 1]])], axis=0)
    Rt_t = np.concatenate([R_t, t_t], axis=1)
    Rt_t = np.concatenate([Rt_t, np.array([[0, 0, 0, 1]])], axis=0)
    G_target_src = Rt_t @ np.linalg.pinv(Rt_s)  # src to tgt  4x4

    return G_target_src


def get_inv_homography(K_src, R_src, t_src, K_tgt, R_tgt, t_tgt, depth):
    """

    Args:
        src_parm: src_parm['K']  3x3   src_parm['R'] 3x3   src_parm['T']  3x1
        tar_parm:
        depth:

    Returns:

    """
    # set source camera coordinates as the world coordinates
    # K_s, R_s, t_s = src_parm['K'], src_parm['R'], src_parm['T']
    # K_t, R_t, t_t = tar_parm['K'], tar_parm['R'], tar_parm['T']
    Rt_src = np.concatenate([R_src, t_src], axis=1)
    Rt_src = np.concatenate([Rt_src, np.array([[0, 0, 0, 1]])], axis=0)
    Rt_tgt = np.concatenate([R_tgt, t_tgt], axis=1)
    Rt_tgt = np.concatenate([Rt_tgt, np.array([[0, 0, 0, 1]])], axis=0)
    G_target_src = Rt_tgt @ np.linalg.pinv(Rt_src)  # target to src  4x4
    # compute inverse homography
    R = G_target_src[:3, :3]
    t = G_target_src[:3, 3].reshape(3, 1)
    n = np.matrix([[0, 0, 1]]).reshape(1, -1)
    a = (-1.*depth).numpy()


    # H = K_src @ (R - t @ n.T / (z + 1e-8)) @ np.linalg.pinv(K_tgt)
    H = K_src @ (R.T + (R.T@t@n@R.T)/(a - n@R.T@t + 1e-8)) @ np.linalg.pinv(K_tgt)
    return H


def get_flow_by_homography(img_h, img_w, H):
    x_map, y_map = np.meshgrid(np.arange(img_w), np.arange(img_h))
    x_map, y_map = x_map.reshape(1, -1), y_map.reshape(1, -1)  # [1, HW]
    z_map = np.ones_like(x_map)
    xyz_ori = np.concatenate((x_map, y_map, z_map), axis=0) # [3, HW]
    xyz_warp = H @ xyz_ori
    xyz_warp[0, :] = (xyz_warp[0, :]/(xyz_warp[2, :]+1e-8) - x_map)
    xyz_warp[1, :] = (xyz_warp[1, :]/(xyz_warp[2, :]+1e-8) - y_map)
    xy_flow = np.asarray(xyz_warp[:2, :]).reshape(2, img_h, img_w)
    return xy_flow


def over_composition(mpi):
    img = mpi[:, 0, :3] * 0
    alpha = mpi[:, 0, 3:] * 0 + 1
    for p_index in range(mpi.shape[1]):
        img += mpi[:, p_index, :3] * mpi[:, p_index, 3:] * alpha
        alpha = alpha*(1 - mpi[:, p_index, 3:])
    return img

def volume_rendering_rgb(mpi, depth):
    P = mpi.shape[1]
    output = 0
    alpha = 0
    I = 0
    depth = depth.squeeze()

    for i in range(0, P-1):
        rgb = mpi[:, i, :3, ...]
        sigma = mpi[:, i, 3:, ...]
        delta = torch.abs(depth[i+1] - depth[i])
        alpha += sigma * delta
        T = torch.exp(-alpha)
        I += T * (1 - torch.exp(-sigma * delta)) * rgb

    return I

def volume_rendering_depth(mpi, depth):
    P = mpi.shape[1]
    output = 0
    alpha = 0
    src_depth = 0
    depth = depth.squeeze()

    for i in range(0, P - 1):
        rgb = mpi[:, i, :3, ...]
        sigma = mpi[:, i, 3:, ...]
        delta = torch.abs(depth[i + 1] - depth[i])
        alpha += sigma * delta
        T = torch.exp(-alpha)
        src_depth += T * (1 - torch.exp(-sigma * delta)) * depth[i]

    return src_depth


def mpi_rendering(mpi, flow):
    '''
    mpi: [B, P, 4, H, W]
    flow: [B, P, 2, H, W]
    '''
    B, P, C, H, W = mpi.shape
    mpi = warp(mpi.view(B*P, -1, H, W), flow.view(B*P, -1, H, W))
    mpi = mpi.view(B, P, C, H, W)
    return over_composition(mpi)


def mpi_depth_rendering(mpi, flow, depth_list):
    '''
    mpi: [B, P, 4, H, W]
    flow: [B, P, 2, H, W]
    depth_list: [P]
    '''
    B, P, C, H, W = mpi.shape
    mpi = warp(mpi.view(B*P, -1, H, W), flow.view(B*P, -1, H, W))
    mpi = mpi.view(B, P, C, H, W)
    depth = mpi[:, 0, 0:1] * 0
    alpha = mpi[:, 0, 3:] * 0 + 1
    for p_index in range(P):
        depth += depth_list[p_index] * mpi[:, p_index, 3:] * alpha
        alpha = alpha*(1 - mpi[:, p_index, 3:])
    return depth

def mpi_volume_rendering(mpi, flow, depth_list):
    """

    Args:
        mpi: [B, P, 4, H, W]
        flow: [B, P, 2, H, W]
        depth_list: [P]

    Returns:

    """
    B, P, C, H, W = mpi.shape
    mpi = warp(mpi.view(B * P, -1, H, W), flow.view(B * P, -1, H, W))
    mpi = mpi.view(B, P, C, H, W)
    return mpi

if __name__ == '__main__':
    from skimage import io
    import json
    import numpy as np
    json_path = '/root/project/datasets/2011_09_26_drive_0001/kitti.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    l_img_meta, r_img_meta = data[0]
    l_img_path = l_img_meta['path'].replace(r'D:\datasets\KITTIRaw_city\city', '/root/project/datasets').replace('\\', '/')
    r_img_path = r_img_meta['path'].replace(r'D:\datasets\KITTIRaw_city\city', '/root/project/datasets').replace('\\', '/')
    l_img = io.imread(l_img_path)
    r_img = io.imread(r_img_path)
    print(type(l_img))
    K_l = np.array(l_img_meta['K']).reshape(3, 3)
    K_r = np.array(r_img_meta['K']).reshape(3, 3)
    R_l = np.array(l_img_meta['R']).reshape(3, 3)
    R_r = np.array(r_img_meta['R']).reshape(3, 3)
    t_l = np.array(l_img_meta['T']).reshape(3, 1)
    t_r = np.array(r_img_meta['T']).reshape(3, 1)
    img_size = l_img.shape
    size_H, size_W, C = img_size
    H = get_inv_homography(K_l, R_l, t_l, K_r, R_r, t_r, 50)
    flow = get_flow_by_homography(img_size[0], img_size[1], H)
    l_img = torch.from_numpy(l_img).permute(2, 0, 1).view(1, C, size_H, size_W)
    flow = torch.from_numpy(flow).view(1, -1, size_H, size_W)
    img_warped = warp(l_img.float(), flow.float()).permute(0, 2, 3, 1).squeeze()


    img_warped = img_warped.numpy()
    l_img = l_img.permute(0, 2, 3, 1).squeeze().numpy()
    io.imsave('/root/project/datasets/2011_09_26_drive_0001/test1.jpg', np.concatenate((l_img, r_img, img_warped), axis=1))
