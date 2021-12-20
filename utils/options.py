# -*- coding: utf-8 -*-

import yaml
import argparse
import os.path as osp
import random
from copy import deepcopy
from os import path as osp
from collections import OrderedDict
from .utils.dist_utils import init_dist, get_dist_info
from .utils.misc import set_random_seed


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class Options:
    @staticmethod
    def parse_options():
        parser = argparse.ArgumentParser(description="Training SR Model")
        parser.add_argument("--opt", default='options/train/smfa_config.yml', type=str)
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--debug', type=bool, default=False)
        args = parser.parse_args()
        opt = Options.parse(args.opt, is_train=True)

        # distributed settings
        if opt['dist']:
            init_dist()
        else:
            print('Disable Distributed Training', flush=True)

        opt['rank'], opt['world_size'] = get_dist_info()

        # random seed
        seed = opt.get('manual_seed')
        if seed is None:
            seed = random.randint(1, 10000)
            opt['manual_seed'] = seed

        set_random_seed(seed + opt['rank'])

        return opt

    @staticmethod
    def ordered_yaml():
        """Support OrderedDict for yaml.

        Returns:
            yaml Loader and Dumper.
        """
        try:
            from yaml import CDumper as Dumper
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Dumper, Loader

        _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

        def dict_representer(dumper, data):
            return dumper.represent_dict(data.items())

        def dict_constructor(loader, node):
            return OrderedDict(loader.construct_pairs(node))

        Dumper.add_representer(OrderedDict, dict_representer)
        Loader.add_constructor(_mapping_tag, dict_constructor)
        return Loader, Dumper

    @staticmethod
    def parse(opt_path, is_train=True):
        with open(opt_path, mode='r') as f:
            yaml_loader, _ = Options.ordered_yaml()
            opt = yaml.load(f, Loader=yaml_loader)

        opt['is_train'] = is_train

        # datasets
        # TODO: Parse datalist file, combine multi train lists and split multi validation lists
        for phase, dataset in opt['datasets'].items():
            # assert (phase in ['train', 'val']), "Only support train and val stages"
            phase = phase.split('_')[0]
            dataset['phase'] = phase
            if 'scale' in opt:
                dataset['scale'] = opt['scale']

        # path
        if is_train:
            exp_root = osp.join(opt['path']['experiments_root'], opt['name'])
            opt['path']['experiments_root'] = exp_root
            opt['path']['ckpts'] = osp.join(exp_root, 'ckpts')
            opt['path']['log'] = osp.join(exp_root, 'log')
            opt['path']['option'] = osp.join(exp_root, 'option')
            opt['path']['tensorboard'] = osp.join(exp_root, 'tensorboard')
            opt['path']['training_states'] = osp.join(exp_root, 'training_states')
            opt['path']['visualization'] = osp.join(exp_root, 'visualization')
            opt['path']['web'] = osp.join(exp_root, 'web')

        # debug
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8

            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8

        return opt


class EvaluationOptions:
    @staticmethod
    def parse_options():
        parser = argparse.ArgumentParser(description="Evaluating SR Model")
        parser.add_argument("--train_opt", default='options/train/EDSR/train_EDSR_Mx2_lisiyi.yml', type=str)
        parser.add_argument("--eval_opt", default='options/eval/EDSR/eval_EDSR_Mx2_lisiyi.yml', type=str)
        parser.add_argument("--direct_mode", default=False, type=str2bool,
                            help="Set true if predicted hr and gt already exist!")
        parser.add_argument("--src_hr", default=None, type=str,
                            help="path to gt HR, used in direct mode")
        parser.add_argument("--pred_hr", default=None, type=str,
                            help="path to predicted HR, used in direct mode")
        parser.add_argument("--test_y_channel", default=False, type=str2bool,
                            help="only test y channel during evaluation stage")
        parser.add_argument("--num_crop_board", default=2, type=int,
                            help="number of board to crop when calculating metrics")
        parser.add_argument("--comp_resize", default=False, type=str2bool,
                            help="Set true to compare HR with resized LR")
        parser.add_argument("--local_rank", type=int, default=0)
        args = parser.parse_args()

        if args.direct_mode:
            opt = None
        else:
            opt_train = Options.parse(args.train_opt, is_train=False)
            opt = Options.parse(args.eval_opt, is_train=False)

            # distributed settings
            if opt['dist']:
                init_dist()
            else:
                print('Disable Distributed Training', flush=True)

            opt['rank'], opt['world_size'] = get_dist_info()

            for key in ['network_g', 'model_type']:
                opt[key] = deepcopy(opt_train[key])

        opt_extra = OrderedDict(args.__dict__)

        return opt, opt_extra





