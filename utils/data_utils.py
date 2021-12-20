# -*- coding: utf-8 -*-

import copy
import os
import os.path as osp
import numpy as np
from glob import glob
from .misc import scandir
from collections import OrderedDict


def paired_paths_from_meta_info_file(folders, keys, meta_info_file,
                                     filename_tmpl):
    """Generate paired paths from an meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        meta_info_file (str): Path to the meta information file.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    with open(meta_info_file, 'r') as fin:
        gt_names = [line.split(' ')[0] for line in fin]

    paths = []
    for gt_name in gt_names:
        basename, ext = osp.splitext(osp.basename(gt_name))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        gt_path = osp.join(gt_folder, gt_name)
        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path)]))
    return paths


def paired_paths_from_lmdb(folders, keys):
    """Generate paired paths from lmdb files.

    Contents of lmdb. Taking the `lq.lmdb` for example, the file structure is:

    lq.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records
    1)image name (with extension),
    2)image shape,
    3)compression level, separated by a white space.
    Example: `baboon.png (120,125,3) 1`

    We use the image name without extension as the lmdb key.
    Note that we use the same key for the corresponding lq and gt images.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
            Note that this key is different from lmdb keys.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith('.lmdb') and gt_folder.endswith('.lmdb')):
        raise ValueError(
            f'{input_key} folder and {gt_key} folder should both in lmdb '
            f'formats. But received {input_key}: {input_folder}; '
            f'{gt_key}: {gt_folder}')
    # ensure that the two meta_info files are the same
    with open(osp.join(input_folder, 'meta_info.txt')) as fin:
        input_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(osp.join(gt_folder, 'meta_info.txt')) as fin:
        gt_lmdb_keys = [line.split('.')[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(
            f'Keys in {input_key}_folder and {gt_key}_folder are different.')
    else:
        paths = []
        for lmdb_key in sorted(input_lmdb_keys):
            paths.append(
                dict([(f'{input_key}_path', lmdb_key),
                      (f'{gt_key}_path', lmdb_key)]))
        return paths


def paired_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    assert len(input_paths) == len(gt_paths), (
        f'{input_key} and {gt_key} datasets have different number of images: '
        f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []
    for gt_path in gt_paths:
        basename, ext = osp.splitext(osp.basename(gt_path))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        assert input_name in input_paths, (f'{input_name} is not in '
                                           f'{input_key}_paths.')
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path)]))
    return paths


def paired_paths_from_datalist(datalist, keys):
    # The len of datalist with each line format at [input.txt + "\t" + gt.txt].
    # recommand command: to combine each line of 2 list with separator \t
    #   paste -d \\t input_list gt_list > datalist

    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')

    input_key, gt_key = keys
    with open(datalist, 'r') as fin:
        paired_list = fin.read().strip().split('\n')

    paths = []
    for idx, item in enumerate(paired_list):
        input_path = item.split()[0]
        gt_path = item.split()[1]

        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path)]))

    return paths


def arbitrary_paths_from_datalist(datalist, keys):
    # The len of datalist with each line format at [input.txt + "\t" + gt.txt].
    # recommand command: to combine each line of 2 list with separator \t
    #   paste -d \\t input_list gt_list > datalist
    with open(datalist, 'r') as fin:
        paired_list = fin.read().strip().split('\n')

    paths = []
    for idx, item in enumerate(paired_list):
        input_paths = item.split()
        assert len(input_paths) == len(keys)

        paths.append(
            dict(
                [(f'{key}_path', input_path)
                    for input_path, key in zip(input_paths, keys)]
                )
            )
    return paths


def construct_unity_camera_parameters(pos, rot, sensize, f, H, W):
    # construct K
    dx = sensize[0]/W
    dy = sensize[1]/H
    K = np.matrix([[f/dx, 0, W/2.],
                [0, f/dy, H/2.],
                [0, 0, 1]]).astype(np.float32)
    # construct R
    R_x = np.matrix([
                        [1, 0, 0],
                        [0, np.cos(rot[0]), np.sin(rot[0])],
                        [0, -np.sin(rot[0]), np.cos(rot[0])]
                    ])
    R_y = np.matrix([
                        [np.cos(rot[1]), 0, -np.sin(rot[1])],
                        [0, 1, 0],
                        [np.sin(rot[1]), 0, np.cos(rot[1])]
                    ])
    R_z = np.matrix([
                        [np.cos(rot[2]), -np.sin(rot[2]), 0],
                        [np.sin(rot[2]), np.cos(rot[2]), 0],
                        [0, 0, 1]
                    ])
    R = R_x @ R_y @ R_z
    # construct t
    t = np.matrix(pos).T
    # t[0][0] = -t[0][0]
    t = -R@t
    return {'K': K, 'R': R, 't': t}


def unitypathparser(img_path):
    meta = img_path.split('/')[-1].split('_')
    meta_dict = {}
    for inf in meta:
        key_val = inf.split('=')
        if len(key_val) == 2:
            meta_dict[key_val[0]] = key_val[1]
    # extract inf
    pos = meta_dict['position']
    rot = meta_dict['rotation']
    sensize = meta_dict['sensorSize']
    f = meta_dict['focal']
    HW = meta_dict['HW']
    near = float(meta_dict['near'])
    far = float(meta_dict['far'])
    # string to number
    pos = [float(p) for p in pos.split(',')]
    rot = [float(r)*np.pi/180. for r in rot.split(',')]
    sensize = [float(s) for s in sensize[1:-1].split(',')]
    f = float(f)
    H = float(HW.split(',')[0])
    W = float(HW.split(',')[1])
    cam_parm = construct_unity_camera_parameters(pos, rot, sensize, f, H, W)
    cam_parm['view'] = int(meta_dict['view'])
    depth_parm = {'near': near, 'far': far}
    return meta_dict, cam_parm, depth_parm


def unityparser(dataroot):
    # load data path
    img_list = glob(osp.join(dataroot, '*_img.png'))
    depth_list = glob(osp.join(dataroot, '*_depth.png'))
    assert len(img_list) == len(depth_list)
    img_list.sort()
    depth_list.sort()
    scence = OrderedDict()
    # construct scence
    for img_path, depth_path in zip(img_list, depth_list):
        # parsing
        meta_dict, cam_parm, depth_parm = unitypathparser(img_path)
        if meta_dict['frame'] not in scence:
            scence[meta_dict['frame']] = []
        scence[meta_dict['frame']].append([img_path, cam_parm, depth_path, depth_parm])
    return scence


def unity4cvi(dataroot, viewrange):
    scence = unityparser(dataroot)
    # construct list
    data_list = []
    meta_list = []
    index_base = 0
    for key in scence:
        # add meta data list
        for cur_meta in scence[key]:
            meta_list.append(cur_meta)
        # store mapping
        cur_frame_viewnum = len(scence[key])
        for viewnum in range(viewrange[0], viewrange[1]+1):
            for i in range(cur_frame_viewnum-viewnum+1):
                for j in range(viewnum):
                    # [l_index, r_index, v_index, v_view]
                    data_list.append([i+index_base,
                                      i+viewnum-1+index_base,
                                      i+j+index_base,
                                      (1.*(scence[key][i+j][1]['view'] - scence[key][i][1]['view']) /
                                       scence[key][i+viewnum-1][1]['view'] - scence[key][i][1]['view'])
                                      ])
        index_base += cur_frame_viewnum
    return data_list, meta_list


def unity4mpi(dataroot, viewrange):
    scence = unityparser(dataroot)
    # construct list
    data_list = []
    meta_list = []
    index_base = 0
    for key in scence:
        # add meta data list
        for cur_meta in scence[key]:
            meta_list.append(cur_meta)
        # store mapping
        cur_frame_viewnum = len(scence[key])
        for viewnum in range(viewrange[0], viewrange[1]+1):
            for i in range(cur_frame_viewnum-viewnum+1):
                for j in range(viewnum):
                    for k in range(viewnum):
                        # [l_index, r_index, m_index, v_index]
                        data_list.append([i+index_base,
                                          i+viewnum-1+index_base,
                                          i+j+index_base,
                                          i+k+index_base
                                          ])
        index_base += cur_frame_viewnum
    return data_list, meta_list


def thu_paths_from_folder(dataroot, viewrange, phase='train'):
    def decode_parm_from_txt(path):
        with open(path, 'r') as f:
            lines = f.readlines()
            K_1 = [float(n) for n in lines[1].strip()[2:-1].split()]
            K_2 = [float(n) for n in lines[2].strip()[1:-1].split()]
            K_3 = [float(n) for n in lines[3].strip()[1:-2].split()]
            K = np.matrix([K_1, K_2, K_3])
            P_1 = [float(n) for n in lines[4].strip()[2:-1].split()]
            P_2 = [float(n) for n in lines[5].strip()[1:-1].split()]
            P_3 = [float(n) for n in lines[6].strip()[1:-2].split()]
            P = np.matrix([P_1, P_2, P_3])
            R = P[:3, :]
            t = P[:, 3:]
            return {'K': K, 'R': R, 't': t}

    def constuct_list_from_path(path, index_base):
        txt_list = glob(osp.join(path, '*cam*.txt'))
        img_list = glob(osp.join(path, '*rgb*.png'))
        depth_list = glob(osp.join(path, '*depth*.png'))
        # sort
        txt_list.sort(key=lambda p: int(p.split('_')[-1].split('.')[0]))
        img_list.sort(key=lambda p: int(p.split('_')[-1].split('.')[0]))
        depth_list.sort(key=lambda p: int(p.split('_')[-1].split('.')[0]))
        # parsing
        meta_list = []
        for img_path, depth_path, txt_path in zip(img_list, depth_list, txt_list):
            cam_parm = decode_parm_from_txt(txt_path)
            depth_parm = None
            meta_list.append([img_path, cam_parm, depth_path, depth_parm])
        # construct data list
        data_list = []
        imgnum = len(txt_list)
        for viewnum in range(viewrange[0], viewrange[1]+1):
            for i in range(imgnum):
                for j in range(viewnum):
                    # [l_index, r_index, v_index, v_view]
                    data_list.append([
                        i+index_base,
                        (i+viewnum-1) % imgnum+index_base,
                        (i+j) % imgnum+index_base,
                        1.*j/(viewnum-1)
                    ])
        return data_list, meta_list

    dir_list = []
    for dir in os.listdir(dataroot):
        if '.' not in dir:
            dir_list.append(osp.join(dataroot, dir))
    dir_list.sort()
    if phase == 'train':
        # dir_list = dir_list[:1]
        dir_list = dir_list[:len(dir_list)//10*9]
    else:
        # dir_list = dir_list[1:]
        dir_list = dir_list[len(dir_list)//10*9:]
    data_list = []
    meta_list = []
    for path in dir_list:
        index_base = len(meta_list)
        cur_data_list, cur_meta_list = constuct_list_from_path(path, index_base)
        data_list += cur_data_list
        meta_list += cur_meta_list
    return data_list, meta_list
