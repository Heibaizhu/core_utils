import torch
import torch.nn as nn


backwarp_tenGrid = {}


def warp(tenInput, tenFlow):
    device = tenFlow.device
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0)) * 2,
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0)) * 2], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)



if __name__ == '__main__':
    from skimage import io
    import json
    import numpy as np
    from mpi_utils import get_mpi_flow, get_inv_homography

    with open(r'/root/project/datasets/SRRandom12/train_unity_triplet.json', 'r') as f:
        data_list = json.load(f)

    datal = data_list[2][0]
    datam = data_list[2][1]
    datar = data_list[2][2]

    img_l = io.imread(
        datal[0].replace('D:\\datasets\\SRRandom12', '/root/project/datasets/SRRandom12/').replace('\\', '/'))
    dep_l = io.imread(
        datal[2].replace('D:\\datasets\\SRRandom12', '/root/project/datasets/SRRandom12/').replace('\\', '/'))
    dep_l = ((dep_l / 255.) ** 4) * (8 - 0.3) + 0.3
    dep_l = dep_l[:, :, 0]
    K_l = np.array(datal[1]['K'])
    R_l = np.array(datal[1]['R'])
    t_l = np.array(datal[1]['t'])

    img_m = io.imread(
        datam[0].replace('D:\\datasets\\SRRandom12', '/root/project/datasets/SRRandom12/').replace('\\', '/'))
    dep_m = io.imread(
        datam[2].replace('D:\\datasets\\SRRandom12', '/root/project/datasets/SRRandom12/').replace('\\', '/'))
    dep_m = ((dep_m / 255.) ** 4) * (8 - 0.3) + 0.3
    K_m = np.array(datam[1]['K'])
    R_m = np.array(datam[1]['R'])
    t_m = np.array(datam[1]['t'])

    img_r = io.imread(
        datar[0].replace('D:\\datasets\\SRRandom12', '/root/project/datasets/SRRandom12/').replace('\\', '/'))
    dep_r = io.imread(
        datar[2].replace('D:\\datasets\\SRRandom12', '/root/project/datasets/SRRandom12/').replace('\\', '/'))
    dep_r = ((dep_r / 255.) ** 4) * (8 - 0.3) + 0.3
    K_r = np.array(datar[1]['K'])  # 3x3
    R_r = np.array(datar[1]['R'])  # 3x3
    t_r = np.array(datar[1]['t'])  # 3x1

    G_l = np.concatenate((R_l, t_l), axis=1)
    G_l = np.concatenate((G_l, np.array([[0, 0, 0, 1]])), axis=0)

    G_m = np.concatenate((R_m, t_m), axis=1)
    G_m = np.concatenate((G_m, np.array([[0, 0, 0, 1]])), axis=0)

    G_r = np.concatenate((R_r, t_r), axis=1)
    G_r = np.concatenate((G_r, np.array([[0, 0, 0, 1]])), axis=0)

    G_m_l = G_m @ np.linalg.inv(G_l)
    K_l_inv = np.linalg.inv(K_l)

    H, W, C = img_l.shape
    B = H * W
    dep = dep_l.flatten()
    N = dep.size
    # flow_list = get_mpi_flow()
    x_map, y_map = np.meshgrid(np.arange(W), np.arange(H))
    x_map, y_map = x_map.reshape(-1, 1, 1), y_map.reshape(-1, 1, 1)  # [HW, 1, 1]
    z_map = np.ones_like(x_map)
    xyz_ori = np.concatenate((x_map, y_map, z_map), axis=1)  # [HW, 3, 1]
    Hs = []
    for depth in dep:
        H = get_inv_homography(K_l, R_l, t_l, K_m, R_m, t_m, torch.tensor(depth))
        Hs.append(H)
    Hs = np.array(Hs) #Nx3x3
    # Hs = np.load('Hs.npy')



    # tensor mutual
    xyz_ori = torch.from_numpy(xyz_ori).float()
    Hs = torch.from_numpy(Hs).float()
    xyz_warp = torch.matmul(Hs, xyz_ori) #HWx3x1
    xyz_warp[:, 0, :] = (xyz_warp[:, 0, :] / (xyz_warp[:, -1, :] + 1e-8) - x_map.squeeze(1))
    xyz_warp[:, 1, :] = (xyz_warp[:, 1, :] / (xyz_warp[:, -1, :] + 1e-8) - y_map.squeeze(1))
    xyz_warp = xyz_warp.squeeze().permute(1, 0).numpy() #3xHW
    H, W, C = img_l.shape
    xy_flow = np.asarray(xyz_warp[:2, :]).reshape(2, H, W)

    #warp
    input_tensor = torch.from_numpy(img_l).permute(2, 0, 1).unsqueeze(0)
    flow_tensor = torch.from_numpy(xy_flow).unsqueeze(0)
    warp_tensor = warp(input_tensor.float(), flow_tensor)
    warp_img = warp_tensor.squeeze().permute(1, 2, 0)
    io.imsave('new_warped_img.png', warp_img)
    io.imsave('new_src_img.png', img_l)
    io.imsave('new_dst_img.png', img_m)




